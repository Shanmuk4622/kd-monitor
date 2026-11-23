-- Run once in the SQL Editor. Copy the four result values into the matching Kaggle secrets.
-- Repeat runs only issue credentials for previously unprovisioned workers.
with generated as materialized (
  select w::smallint as worker_id, replace(gen_random_uuid()::text,'-','') ||
         replace(gen_random_uuid()::text,'-','') as token from generate_series(0,3) w
), added as (
  insert into kd_private.worker_keys(worker_id,token_hash)
  select worker_id,sha256(convert_to(token,'UTF8')) from generated
  on conflict do nothing returning worker_id
)
select 'KD_WORKER_TOKEN_' || g.worker_id as kaggle_secret, g.token as secret_value
from generated g join added using(worker_id) order by g.worker_id;
