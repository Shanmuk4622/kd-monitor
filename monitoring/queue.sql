-- Run once in the existing project's Supabase SQL Editor. Safe to rerun during training.
begin;
create table if not exists public.kd_jobs (
    run_id text primary key, original_worker smallint not null, model text not null,
    teacher text, bank text, updates integer not null, sequence integer not null,
    state text not null default 'legacy' check (state in ('legacy','pending','claimed','audited')),
    resumable boolean not null default false, bank_ready boolean not null default false,
    worker smallint, session_id uuid, ticket uuid, expires timestamptz,
    estimate_seconds double precision, next_check timestamptz
);
alter table public.kd_jobs enable row level security;
revoke all on public.kd_jobs from public,anon,authenticated;
grant select(run_id,original_worker,model,teacher,bank,updates,sequence,state,resumable,bank_ready,worker,expires)
    on public.kd_jobs to authenticated;
drop policy if exists kd_viewer on public.kd_jobs;
create policy kd_viewer on public.kd_jobs for select to authenticated
using ((select auth.jwt())->'app_metadata'->>'kd_monitor'='viewer');

-- Frozen campaign inventory. The original worker is used only to migrate its legacy fits.
insert into public.kd_jobs(run_id,original_worker,model,teacher,bank,updates,sequence) values
('kaggle-v2.cifar100.resnet34.teacher.907',0,'resnet34',null,null,31680,1),
('kaggle-v2.cifar100.resnet18.hard_labels.42',3,'resnet18',null,null,21120,2),
('kaggle-v2.cifar100.resnet18.disagreement.42',0,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.42.disagreement',21120,3),
('kaggle-v2.cifar100.resnet18.entropy.42',0,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.42.entropy',21120,4),
('kaggle-v2.cifar100.resnet18.uniform.42',0,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.42.uniform',21120,5),
('kaggle-v2.cifar100.resnet18.shuffled.42',0,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.42.shuffled',21120,6),
('kaggle-v2.cifar100.resnet18.live_kd.42',0,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907',null,21120,7),
('kaggle-v2.cifar100.resnet18.disagreement.123',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.123.disagreement',21120,8),
('kaggle-v2.cifar100.resnet18.shuffled.123',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.123.shuffled',21120,9),
('kaggle-v2.cifar100.resnet18.entropy.123',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.123.entropy',21120,10),
('kaggle-v2.cifar100.resnet18.live_kd.123',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907',null,21120,11),
('kaggle-v2.cifar100.resnet18.hard_labels.123',3,'resnet18',null,null,21120,12),
('kaggle-v2.cifar100.resnet18.uniform.123',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.123.uniform',21120,13),
('kaggle-v2.cifar100.resnet18.disagreement.2026',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.2026.disagreement',21120,14),
('kaggle-v2.cifar100.resnet18.uniform.2026',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.2026.uniform',21120,15),
('kaggle-v2.cifar100.resnet18.entropy.2026',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.2026.entropy',21120,16),
('kaggle-v2.cifar100.resnet18.live_kd.2026',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907',null,21120,17),
('kaggle-v2.cifar100.resnet18.shuffled.2026',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.2026.shuffled',21120,18),
('kaggle-v2.cifar100.resnet18.hard_labels.2026',3,'resnet18',null,null,21120,19),
('kaggle-v2.cifar100.resnet18.disagreement.31415',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.31415.disagreement',21120,20),
('kaggle-v2.cifar100.resnet18.shuffled.31415',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.31415.shuffled',21120,21),
('kaggle-v2.cifar100.resnet18.hard_labels.31415',3,'resnet18',null,null,21120,22),
('kaggle-v2.cifar100.resnet18.uniform.31415',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.31415.uniform',21120,23),
('kaggle-v2.cifar100.resnet18.live_kd.31415',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907',null,21120,24),
('kaggle-v2.cifar100.resnet18.entropy.31415',3,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.31415.entropy',21120,25),
('kaggle-v2.cifar100.resnet18.entropy.27182',2,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.27182.entropy',21120,26),
('kaggle-v2.cifar100.resnet18.disagreement.27182',2,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.27182.disagreement',21120,27),
('kaggle-v2.cifar100.resnet18.uniform.27182',2,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.27182.uniform',21120,28),
('kaggle-v2.cifar100.resnet18.shuffled.27182',2,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.27182.shuffled',21120,29),
('kaggle-v2.cifar100.resnet18.live_kd.27182',2,'resnet18','kaggle-v2.cifar100.resnet34.teacher.907',null,21120,30),
('kaggle-v2.cifar100.resnet18.hard_labels.27182',3,'resnet18',null,null,21120,31),
('kaggle-v2.cifar100.mobilenetv4_conv_small.disagreement.42',0,'mobilenetv4_conv_small','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.42.disagreement',21120,32),
('kaggle-v2.cifar100.mobilenetv4_conv_small.uniform.42',0,'mobilenetv4_conv_small','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.42.uniform',21120,33),
('kaggle-v2.cifar100.mobilenetv4_conv_small.shuffled.42',0,'mobilenetv4_conv_small','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.42.shuffled',21120,34),
('kaggle-v2.cifar100.mobilenetv4_conv_small.disagreement.123',3,'mobilenetv4_conv_small','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.123.disagreement',21120,35),
('kaggle-v2.cifar100.mobilenetv4_conv_small.uniform.123',3,'mobilenetv4_conv_small','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.123.uniform',21120,36),
('kaggle-v2.cifar100.mobilenetv4_conv_small.shuffled.123',3,'mobilenetv4_conv_small','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.123.shuffled',21120,37),
('kaggle-v2.cifar100.mobilenetv4_conv_small.disagreement.2026',3,'mobilenetv4_conv_small','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.2026.disagreement',21120,38),
('kaggle-v2.cifar100.mobilenetv4_conv_small.shuffled.2026',3,'mobilenetv4_conv_small','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.2026.shuffled',21120,39),
('kaggle-v2.cifar100.mobilenetv4_conv_small.uniform.2026',3,'mobilenetv4_conv_small','kaggle-v2.cifar100.resnet34.teacher.907','kaggle-v2.cifar100.resnet34.2026.uniform',21120,40),
('kaggle-v2.cifar10.resnet34.teacher.907',1,'resnet34',null,null,31680,41),
('kaggle-v2.cifar10.resnet18.uniform.42',1,'resnet18','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.42.uniform',21120,42),
('kaggle-v2.cifar10.resnet18.disagreement.42',1,'resnet18','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.42.disagreement',21120,43),
('kaggle-v2.cifar10.resnet18.shuffled.42',1,'resnet18','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.42.shuffled',21120,44),
('kaggle-v2.cifar10.resnet18.shuffled.123',1,'resnet18','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.123.shuffled',21120,45),
('kaggle-v2.cifar10.resnet18.disagreement.123',1,'resnet18','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.123.disagreement',21120,46),
('kaggle-v2.cifar10.resnet18.uniform.123',1,'resnet18','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.123.uniform',21120,47),
('kaggle-v2.cifar10.resnet18.disagreement.2026',2,'resnet18','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.2026.disagreement',21120,48),
('kaggle-v2.cifar10.resnet18.uniform.2026',2,'resnet18','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.2026.uniform',21120,49),
('kaggle-v2.cifar10.resnet18.shuffled.2026',2,'resnet18','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.2026.shuffled',21120,50),
('kaggle-v2.tiny_imagenet200.resnet34.teacher.907',2,'resnet34',null,null,63360,51),
('kaggle-v2.tiny_imagenet200.resnet18.uniform.42',2,'resnet18','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.42.uniform',42240,52),
('kaggle-v2.tiny_imagenet200.resnet18.disagreement.42',2,'resnet18','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.42.disagreement',42240,53),
('kaggle-v2.tiny_imagenet200.resnet18.shuffled.42',2,'resnet18','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.42.shuffled',42240,54),
('kaggle-v2.tiny_imagenet200.resnet18.shuffled.123',0,'resnet18','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.123.shuffled',42240,55),
('kaggle-v2.tiny_imagenet200.resnet18.disagreement.123',0,'resnet18','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.123.disagreement',42240,56),
('kaggle-v2.tiny_imagenet200.resnet18.uniform.123',0,'resnet18','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.123.uniform',42240,57),
('kaggle-v2.tiny_imagenet200.resnet18.shuffled.2026',1,'resnet18','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.2026.shuffled',42240,58),
('kaggle-v2.tiny_imagenet200.resnet18.disagreement.2026',1,'resnet18','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.2026.disagreement',42240,59),
('kaggle-v2.tiny_imagenet200.resnet18.uniform.2026',1,'resnet18','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.2026.uniform',42240,60),
('kaggle-v2.cifar10.mobilenetv4_conv_small.disagreement.42',1,'mobilenetv4_conv_small','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.42.disagreement',21120,61),
('kaggle-v2.cifar10.mobilenetv4_conv_small.uniform.42',1,'mobilenetv4_conv_small','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.42.uniform',21120,62),
('kaggle-v2.cifar10.mobilenetv4_conv_small.shuffled.42',1,'mobilenetv4_conv_small','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.42.shuffled',21120,63),
('kaggle-v2.cifar10.mobilenetv4_conv_small.uniform.123',1,'mobilenetv4_conv_small','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.123.uniform',21120,64),
('kaggle-v2.cifar10.mobilenetv4_conv_small.disagreement.123',1,'mobilenetv4_conv_small','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.123.disagreement',21120,65),
('kaggle-v2.cifar10.mobilenetv4_conv_small.shuffled.123',1,'mobilenetv4_conv_small','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.123.shuffled',21120,66),
('kaggle-v2.cifar10.mobilenetv4_conv_small.uniform.2026',2,'mobilenetv4_conv_small','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.2026.uniform',21120,67),
('kaggle-v2.cifar10.mobilenetv4_conv_small.shuffled.2026',2,'mobilenetv4_conv_small','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.2026.shuffled',21120,68),
('kaggle-v2.cifar10.mobilenetv4_conv_small.disagreement.2026',2,'mobilenetv4_conv_small','kaggle-v2.cifar10.resnet34.teacher.907','kaggle-v2.cifar10.resnet34.2026.disagreement',21120,69),
('kaggle-v2.tiny_imagenet200.mobilenetv4_conv_small.disagreement.42',2,'mobilenetv4_conv_small','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.42.disagreement',42240,70),
('kaggle-v2.tiny_imagenet200.mobilenetv4_conv_small.uniform.42',2,'mobilenetv4_conv_small','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.42.uniform',42240,71),
('kaggle-v2.tiny_imagenet200.mobilenetv4_conv_small.shuffled.42',2,'mobilenetv4_conv_small','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.42.shuffled',42240,72),
('kaggle-v2.tiny_imagenet200.mobilenetv4_conv_small.shuffled.123',0,'mobilenetv4_conv_small','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.123.shuffled',42240,73),
('kaggle-v2.tiny_imagenet200.mobilenetv4_conv_small.uniform.123',0,'mobilenetv4_conv_small','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.123.uniform',42240,74),
('kaggle-v2.tiny_imagenet200.mobilenetv4_conv_small.disagreement.123',0,'mobilenetv4_conv_small','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.123.disagreement',42240,75),
('kaggle-v2.tiny_imagenet200.mobilenetv4_conv_small.shuffled.2026',1,'mobilenetv4_conv_small','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.2026.shuffled',42240,76),
('kaggle-v2.tiny_imagenet200.mobilenetv4_conv_small.disagreement.2026',1,'mobilenetv4_conv_small','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.2026.disagreement',42240,77),
('kaggle-v2.tiny_imagenet200.mobilenetv4_conv_small.uniform.2026',1,'mobilenetv4_conv_small','kaggle-v2.tiny_imagenet200.resnet34.teacher.907','kaggle-v2.tiny_imagenet200.resnet34.2026.uniform',42240,78)
on conflict(run_id) do nothing;

create or replace function public.kd_dispatch(p_worker integer,p_token text,p_session uuid,
    p_action text,p_data jsonb default '{}') returns jsonb
language plpgsql security definer set search_path='' as $$
declare
    job public.kd_jobs%rowtype;
    now_at timestamptz;
    ttl interval := interval '5 minutes';
begin
    if p_session is null or not exists(select 1 from kd_private.worker_keys k where k.worker_id=p_worker
        and k.token_hash=sha256(convert_to(p_token,'UTF8'))) then
        raise insufficient_privilege using message='Invalid worker credential';
    end if;
    -- Serialize cross-row decisions (including paired jobs sharing a prediction bank).
    perform pg_advisory_xact_lock(4622,78);
    now_at := clock_timestamp();
    if p_action='ping' then
        return jsonb_build_object('migration_required',exists(select 1 from public.kd_jobs
            where state='legacy' and original_worker=p_worker));
    elsif p_action='migrate' then
        update public.kd_jobs j set state='audited',worker=j.original_worker
          where state='legacy' and (original_worker=p_worker or model='resnet34')
            and (p_data->'audited') ? j.run_id;
        update public.kd_jobs j set state='pending',resumable=(p_data->'resumable') ? j.run_id
          where state='legacy' and original_worker=p_worker;
        update public.kd_jobs j set estimate_seconds=(p_data->'estimates'->>j.run_id)::double precision
          where original_worker=p_worker and (p_data->'estimates') ? j.run_id;
        update public.kd_jobs j set bank_ready=true where not bank_ready and exists
          (select 1 from public.kd_jobs a where a.bank=j.bank and a.state='audited');
        return '{}';
    elsif p_action='observe_teacher' then
        update public.kd_jobs set state='audited',worker=original_worker
          where state='legacy' and model='resnet34' and run_id=p_data->>'run';
        return '{}';
    elsif p_action='claim' then
        update public.kd_jobs set state='pending',worker=null,session_id=null,ticket=null,expires=null
           where state='claimed' and expires<=now_at;
        select * into job from public.kd_jobs where state='claimed' and worker=p_worker and session_id=p_session;
        if not found then
            select j.* into job from public.kd_jobs j where j.state='pending'
              and (jsonb_array_length(p_data->'models')=0 or (p_data->'models') ? j.model
                   or exists(select 1 from public.kd_jobs s where s.teacher=j.run_id and s.state<>'audited'
                              and (p_data->'models') ? s.model))
              and (j.teacher is null or exists(select 1 from public.kd_jobs t
                  where t.run_id=j.teacher and t.state='audited'))
              and (j.bank_ready or not exists(select 1 from public.kd_jobs b
                  where b.bank=j.bank and b.state='claimed'))
              order by (j.model='resnet34') desc, j.resumable desc,
                (j.bank is not null and not j.bank_ready and exists(select 1 from public.kd_jobs b
                  where b.bank=j.bank and b.run_id<>j.run_id and b.state='pending')) desc,
                coalesce(j.estimate_seconds,j.updates) desc, j.sequence limit 1;
            if not found then
                if not exists(select 1 from public.kd_jobs j where state<>'audited'
                      and (jsonb_array_length(p_data->'models')=0 or (p_data->'models') ? j.model))
                    then return '{"finished":true}'; end if;
                select t.* into job from public.kd_jobs t where t.state='legacy' and t.model='resnet34'
                    and (t.next_check is null or t.next_check<=now_at)
                    and exists(select 1 from public.kd_jobs s where s.teacher=t.run_id and s.state='pending')
                    order by t.sequence limit 1;
                if found then
                    update public.kd_jobs set next_check=now_at+interval '3 minutes' where run_id=job.run_id;
                    return jsonb_build_object('waiting',true,'check_teacher',job.run_id);
                end if;
                return '{"waiting":true}';
            end if;
            update public.kd_jobs set worker=p_worker,session_id=p_session,ticket=gen_random_uuid(),
                expires=now_at+ttl,state='claimed' where run_id=job.run_id returning * into job;
        end if;
        return jsonb_build_object('run',job.run_id,'ticket',job.ticket,'resumable',job.resumable);
    end if;
    select * into job from public.kd_jobs where run_id=p_data->>'run' and worker=p_worker
        and session_id=p_session and ticket=(p_data->>'ticket')::uuid;
    if not found then raise exception 'Job lease lost'; end if;
    if p_action='finish' and job.state='audited' then return '{}'; end if;
    if job.state<>'claimed' or job.expires<=now_at then raise exception 'Job lease expired'; end if;
    if p_action='renew' then
        if p_data ? 'sample' then
            if (p_data->'sample'->>'worker_id')::integer is distinct from p_worker then
                raise exception 'Wrong worker in queue sample';
            end if;
            perform public.kd_report(p_token,p_data->'sample');
        end if;
        update public.kd_jobs set expires=now_at+ttl where run_id=job.run_id;
    elsif p_action='checkpoint' then
        update public.kd_jobs set resumable=true where run_id=job.run_id and not resumable;
    elsif p_action='bank' then
        update public.kd_jobs set bank_ready=true where bank=job.bank and not bank_ready;
    elsif p_action='finish' then
        update public.kd_jobs set state='audited',expires=null where run_id=job.run_id;
    elsif p_action='yield' then
        update public.kd_jobs set state='pending',worker=null,expires=null,ticket=null,session_id=null,
            resumable=resumable or coalesce((p_data->>'resumable')::boolean,false) where run_id=job.run_id;
    else raise exception 'Unknown queue action'; end if;
    return '{}';
end $$;
revoke all on function public.kd_dispatch(integer,text,uuid,text,jsonb) from public,anon,authenticated;
grant execute on function public.kd_dispatch(integer,text,uuid,text,jsonb) to anon;
commit;
