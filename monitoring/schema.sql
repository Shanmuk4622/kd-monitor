-- Run the whole file in the Supabase SQL Editor. Re-running preserves worker credentials.
begin;
create schema if not exists kd_private;
revoke all on schema kd_private from public, anon, authenticated;
create table if not exists kd_private.worker_keys (
    worker_id smallint primary key check (worker_id between 0 and 3),
    token_hash bytea not null
);
revoke all on kd_private.worker_keys from public, anon, authenticated;

create table if not exists public.kd_workers (
    worker_id smallint primary key check (worker_id between 0 and 3),
    received_at timestamptz not null,
    sample jsonb not null
);
create table if not exists public.kd_events (
    id bigint generated always as identity primary key,
    worker_id smallint not null references public.kd_workers,
    recorded_at timestamptz not null,
    status text not null, run_id text, epoch integer, comment text not null
);
create index if not exists kd_events_worker_id_idx on public.kd_events(worker_id, id desc);
alter table public.kd_workers enable row level security;
alter table public.kd_events enable row level security;
revoke all on public.kd_workers, public.kd_events from public, anon, authenticated;
grant select on public.kd_workers, public.kd_events to authenticated;
drop policy if exists kd_viewer on public.kd_workers;
create policy kd_viewer on public.kd_workers for select to authenticated
using ((select auth.jwt())->'app_metadata'->>'kd_monitor' = 'viewer');
drop policy if exists kd_viewer on public.kd_events;
create policy kd_viewer on public.kd_events for select to authenticated
using ((select auth.jwt())->'app_metadata'->>'kd_monitor' = 'viewer');

-- Bounded telemetry for existing and upgraded workers alike.
create table if not exists public.kd_trends (
    worker_id smallint not null references public.kd_workers, bucket timestamptz not null,
    sampled_at timestamptz not null, run_id text, loss double precision, accuracy double precision,
    cpu double precision, gpu double precision, primary key(worker_id,bucket)
);
create table if not exists public.kd_run_activity (
    worker_id smallint not null references public.kd_workers, run_id text not null,
    first_seen timestamptz not null, last_seen timestamptz not null,
    observed_seconds double precision not null default 0, epoch integer,
    primary key(worker_id,run_id)
);
alter table public.kd_trends enable row level security;
alter table public.kd_run_activity enable row level security;
revoke all on public.kd_trends,public.kd_run_activity from public,anon,authenticated;
grant select on public.kd_trends,public.kd_run_activity to authenticated;
drop policy if exists kd_viewer on public.kd_trends;
create policy kd_viewer on public.kd_trends for select to authenticated
using ((select auth.jwt())->'app_metadata'->>'kd_monitor'='viewer');
drop policy if exists kd_viewer on public.kd_run_activity;
create policy kd_viewer on public.kd_run_activity for select to authenticated
using ((select auth.jwt())->'app_metadata'->>'kd_monitor'='viewer');

create or replace function kd_private.capture_progress() returns trigger
language plpgsql security definer set search_path='' as $$
declare
    p jsonb := new.sample;
    run text := p->>'run_id';
    ts timestamptz := (p->>'sampled_at')::timestamptz;
    slot timestamptz := date_bin('5 minutes',ts,'2000-01-01 00:00+00'::timestamptz);
    elapsed double precision := 0;
begin
    if tg_op='UPDATE' and old.sample->>'started_at'=p->>'started_at'
        and old.sample->>'run_id'=run and old.sample->>'status'='running' then
        elapsed := extract(epoch from (ts-(old.sample->>'sampled_at')::timestamptz));
        if elapsed not between 0 and 90 then elapsed := 0; end if;
    end if;
    if run ~ '^kaggle-v2\.[a-z0-9_]+\.[a-z0-9_]+\.[a-z_]+\.[0-9]+$' then
        insert into public.kd_run_activity as a values
            (new.worker_id,run,ts,ts,elapsed,(p->>'epoch')::integer)
        on conflict(worker_id,run_id) do update set last_seen=excluded.last_seen,
            observed_seconds=a.observed_seconds+excluded.observed_seconds,
            epoch=coalesce(excluded.epoch,a.epoch);
    end if;
    insert into public.kd_trends values
        (new.worker_id,slot,ts,run,(p->>'loss')::float8,(p->>'accuracy')::float8,
        (p->>'cpu')::float8,(p->'gpus'->0->>'utilization')::float8)
    on conflict(worker_id,bucket) do update set sampled_at=excluded.sampled_at,run_id=excluded.run_id,
        loss=excluded.loss,accuracy=excluded.accuracy,cpu=excluded.cpu,gpu=excluded.gpu;
    delete from public.kd_trends where bucket <= slot-interval '12 hours';
    return new;
end $$;
revoke all on function kd_private.capture_progress() from public, anon, authenticated;
drop trigger if exists kd_capture_progress on public.kd_workers;
create trigger kd_capture_progress after insert or update on public.kd_workers
for each row execute function kd_private.capture_progress();

create or replace function public.kd_report(p_token text, p jsonb) returns void
language plpgsql security definer set search_path = '' as $$
declare
    w smallint := (p->>'worker_id')::smallint;
    now_at timestamptz := clock_timestamp();
    started timestamptz := (p->>'started_at')::timestamptz;
    sampled timestamptz := (p->>'sampled_at')::timestamptz;
    seq_no bigint := (p->>'seq')::bigint;
    prior jsonb;
    last_event timestamptz;
begin
    perform 1 from kd_private.worker_keys k
      where k.worker_id=w and k.token_hash=sha256(convert_to(p_token,'UTF8')) for update;
    if not found then raise insufficient_privilege using message='Invalid worker credential'; end if;
    if jsonb_typeof(p) <> 'object' or octet_length(p::text)>16384
       or not (p ?& array['worker_id','started_at','sampled_at','seq','status','run_id','comment',
                            'model','epoch','loss','accuracy','cpu','gpus'])
       or (p - array['worker_id','started_at','sampled_at','seq','status','run_id','comment',
                     'model','epoch','loss','accuracy','cpu','gpus','details']) <> '{}'::jsonb
       or (p ? 'details' and jsonb_typeof(p->'details') <> 'object')
       or started is null or sampled is null or seq_no is null or seq_no<0
       or sampled < started or sampled > now_at + interval '5 minutes'
       or sampled < now_at - interval '5 minutes'
       or coalesce(p->>'status','') not in ('running','completed','failed','stopped')
       or coalesce(length(p->>'comment'),0) not between 1 and 200
       or jsonb_typeof(p->'gpus') <> 'array' then
        raise exception 'Invalid worker sample';
    end if;
    select sample into prior from public.kd_workers where worker_id=w;
    if prior is not null and (started,seq_no) <=
       ((prior->>'started_at')::timestamptz,(prior->>'seq')::bigint) then return; end if;
    insert into public.kd_workers(worker_id,received_at,sample) values(w,now_at,p)
    on conflict(worker_id) do update set received_at=excluded.received_at,sample=excluded.sample;

    select recorded_at into last_event from public.kd_events where worker_id=w order by id desc limit 1;
    if prior is null or started is distinct from (prior->>'started_at')::timestamptz
       or p->>'run_id' is distinct from prior->>'run_id'
       or p->>'status' is distinct from prior->>'status'
       or p->>'epoch' is distinct from prior->>'epoch'
       or last_event is null or last_event < now_at - interval '15 minutes' then
        insert into public.kd_events(worker_id,recorded_at,status,run_id,epoch,comment)
        values(w,now_at,p->>'status',p->>'run_id',(p->>'epoch')::integer,p->>'comment');
        delete from public.kd_events where worker_id=w and id in
            (select id from public.kd_events where worker_id=w order by id desc offset 40);
    end if;
end;
$$;
revoke all on function public.kd_report(text,jsonb) from public, authenticated, anon;
grant execute on function public.kd_report(text,jsonb) to anon;
do $$
declare t text;
begin
    foreach t in array array['kd_workers','kd_events'] loop
        if not exists (select 1 from pg_publication_tables where pubname='supabase_realtime'
                       and schemaname='public' and tablename=t) then
            execute format('alter publication supabase_realtime add table public.%I',t);
        end if;
    end loop;
end $$;
commit;
