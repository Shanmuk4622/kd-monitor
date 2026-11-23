-- Run in the project owner's SQL Editor only when replacing all four worker credentials.
-- Each old token stops working as soon as this statement commits.
with fresh as materialized (
  select w::smallint as worker_id,
         replace(gen_random_uuid()::text,'-','') || replace(gen_random_uuid()::text,'-','') as token
  from generate_series(0,3) w
), rotated as (
  insert into kd_private.worker_keys(worker_id,token_hash)
  select worker_id,sha256(convert_to(token,'UTF8')) from fresh
  on conflict (worker_id) do update set token_hash=excluded.token_hash
  returning worker_id
)
select 'KD_WORKER_TOKEN_' || f.worker_id as kaggle_secret, f.token as secret_value
from fresh f join rotated using(worker_id) order by f.worker_id;
