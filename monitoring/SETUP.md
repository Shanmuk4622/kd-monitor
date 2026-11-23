# Monitor the four Kaggle workers

Kaggle sends one JSON request per running worker every 30 seconds. Supabase stores the latest sample for each worker and a short event history. A signed-in dashboard subscribes to both tables, fetches an initial snapshot, and refreshes after reconnecting. The website needs only static hosting. After setup, the Supabase Table Editor also works as a simpler manual viewer.

## 1. Get the project URL and public key

Open [Supabase Dashboard](https://supabase.com/dashboard), select your project (or create one), then click **Connect**. Copy the **Project URL**, such as `https://abcdefgh.supabase.co`, and the **publishable key** beginning with `sb_publishable_`. You can also find keys in **Settings → API Keys**. The scripts accept the legacy **anon** key in the same setting. Use the publishable key for a new setup. The browser and notebooks both use that public key. Permission to write comes from a separate random credential for each worker. Keep elevated keys, database passwords and worker tokens out of the website. [Supabase key locations](https://supabase.com/docs/guides/getting-started/api-keys)

## 2. Create the tables and enable Realtime

In the same Supabase project, click **SQL Editor → New query**. Open [schema.sql](schema.sql) in the project, copy the entire file into the query window, and click **Run**. A success message with no result rows is expected. Run the SQL as the project owner through this dashboard. The script enables row-level security and adds both public tables to the `supabase_realtime` publication. If the Data API is disabled in your project, enable it for the `public` schema. Keep `kd_private` outside the exposed schemas. [Realtime publication settings](https://supabase.com/docs/guides/realtime/postgres-changes)

| Database object | Contents and access |
| --- | --- |
| `public.kd_workers` | Primary key `worker_id` (0–3), server `received_at`, and latest JSON `sample`. Authenticated viewers can read. |
| `public.kd_events` | Event ID, worker ID, server time, status, run ID, epoch and comment. Authenticated viewers can read. |
| `kd_private.worker_keys` | Worker ID and SHA-256 hash of the random writer token. Only the project owner and the ingestion function need access. |
| `public.kd_report(p_token, p)` | Checks the token against the reported worker, serializes writes for that worker, ignores older samples and updates status and history in one transaction. |

## 3. Generate the four worker credentials

Choose **SQL Editor → New query** again. Paste [worker_keys.sql](worker_keys.sql) and click **Run**. The Results panel shows four rows named `KD_WORKER_TOKEN_0` through `KD_WORKER_TOKEN_3`. Copy each value into its corresponding Kaggle secret in step 5. Treat the query result as private. Re-running the query preserves existing credentials and returns only newly created ones. If you lose one token, remove just that worker's hash in the SQL Editor, for example `delete from kd_private.worker_keys where worker_id = 2;`, then run the generation query again and update that worker's Kaggle secret.

## 4. Create your dashboard login

In Supabase, open **Authentication → Users → Add user → Create new user**. Enter your email and a password and mark the email as confirmed. If your account already appears in Users, use it. In **SQL Editor → New query**, run the following after replacing the email. A returned row confirms that the correct user received viewer access. Sign out and back in if the dashboard was open before the change.

```sql
update auth.users
set raw_app_meta_data = coalesce(raw_app_meta_data, '{}'::jsonb)
                        || '{"kd_monitor":"viewer"}'::jsonb
where lower(email) = lower('your-email@example.com')
returning id, email;
```

The dashboard uses this Supabase Auth account. Your Kaggle and Supabase Dashboard account passwords are separate from this viewer login.

## 5. Connect the four Kaggle notebooks

Import the four updated `02_Kaggle_Train*.ipynb` files from the project’s `notebooks/` folder. In each Kaggle notebook turn on **Internet**, open **Add-ons → Secrets**, add the applicable name/value pairs, and enable access for that notebook. Each worker uses its preset ID. If the notebooks run under separate Kaggle accounts, add the common settings to each account. [Kaggle Secrets](https://www.kaggle.com/product-feedback/114053)

| Kaggle secret | What to paste | Where |
| --- | --- | --- |
| `SUPABASE_URL` | Project URL from step 1 | All four |
| `SUPABASE_ANON_KEY` | Publishable or legacy anon key from step 1 | All four |
| `KD_WORKER_TOKEN_0` | Worker 0 value from step 3 | `02_Kaggle_Train.ipynb` |
| `KD_WORKER_TOKEN_1` | Worker 1 value | `02_Kaggle_Train_Worker1.ipynb` |
| `KD_WORKER_TOKEN_2` | Worker 2 value | `02_Kaggle_Train_Worker2.ipynb` |
| `KD_WORKER_TOKEN_3` | Worker 3 value | `02_Kaggle_Train_Worker3.ipynb` |
| `HF_TOKEN` | Existing Hugging Face write token | All four, as before |

Once NB01 has qualified, start each training notebook in its own session. For an unattended run use Kaggle’s server-side **Save Version → Save & Run All** option. Avoid starting a second copy of an already-running worker ID. A notebook already executing older code picks up monitoring on its next launch from the updated file; its completed checkpoints can be resumed. The first successful delivery prints `Supabase monitor connected for worker N.` Monitoring begins after the training session acquires its worker lease. NB01 and the results notebook do not occupy worker slots.

## 6. Open the website

The public site configuration for this project is filled in. To use GitHub Pages, create a public dashboard-only repository, upload the four files inside `monitoring/site/` to the repository root, and commit to `main`. Open **Settings → Pages → Build and deployment → Deploy from a branch**. Select **main**, **/(root)**, and **Save**. Open the published URL shown on the Pages screen. Each viewer signs in through Supabase. [GitHub Pages publishing settings](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

On the Windows computer holding the encrypted entries, run `.\monitoring\KaggleSecret.ps1 -Name SUPABASE_URL` in PowerShell from the project folder. It copies one value to the clipboard. Substitute any of the six saved Kaggle names. After rotating a token, add `-Update` to enter its replacement at a hidden prompt. Store the result in Kaggle too. The project owner can replace all four tokens in one transaction using [rotate_worker_keys.sql](rotate_worker_keys.sql) in the Supabase SQL Editor.

Open [site/config.js](site/config.js) in a text editor. Paste the Project URL into `url` and the publishable or anon key into `key`, between the existing quotes. The other website files are [index.html](site/index.html), [style.css](site/style.css) and [app.js](site/app.js). Keep all four files together.

To open it on your own computer, run this in PowerShell **from the KD new project folder**.

```powershell
python -m http.server 8000 --bind 127.0.0.1 --directory monitoring/site
```

Visit [localhost:8000](http://localhost:8000) and sign in with the viewer account from step 4. To use your existing website, copy the contents of `monitoring/site/` to a static folder such as `/kd-monitor/` on that host and visit its HTTPS URL. The site needs access to the pinned Supabase JS library on jsDelivr and to your project's HTTPS and WebSocket endpoints. Closing the dashboard has no effect on the Kaggle jobs.

## 7. Verify the first report

In **Table Editor**, select `kd_workers`, or run this in **SQL Editor → New query**.

```sql
select worker_id, received_at, sample->>'model' as model,
       sample->>'epoch' as epoch, sample->>'loss' as loss, sample->>'comment' as comment
from public.kd_workers order by worker_id;

select recorded_at, worker_id, epoch, comment
from public.kd_events order by id desc limit 20;
```

Start with one worker. Its received time should advance about twice per minute. The other cards say **No report** until their first write. If the tables have rows but the website is blank, check the viewer email in step 4, sign in again and check that the website points to the same project. If rows appear in SQL but live updates are delayed, check **Database → Publications → supabase_realtime** and the connection indicator. The website also fetches a snapshot once per minute.

If the notebook reports HTTP 401 or 403, check the project URL, public key, matching worker token and its Kaggle access toggle. An HTTP 404 can indicate that the schema was installed in another project or the Data API is disabled. Network failures skip that sample; the next heartbeat retries with the latest state. A hard server stop can prevent the terminal event from being delivered. A previously running worker shows **stale** after 90 seconds without a new receipt. This means the dashboard lost contact; Kaggle logs can distinguish a network interruption from a stopped job.

## Payload and metric definitions

[examples.json](examples.json) contains two complete example request bodies for `POST /rest/v1/rpc/kd_report`. They illustrate a training checkpoint and a worker starting up. The actual sender inserts a current timestamp and its Kaggle token, and supplies the public API key in the `apikey` header. Sample timestamps in the examples are illustrative; the server checks freshness.

| Field | Meaning |
| --- | --- |
| `run_id`, `model` | Assigned experiment and model. During setup the model may be null. |
| `epoch`, `loss`, `accuracy` | One-based epoch, mean training objective and top-1 percentage of the last minibatch in the latest locally committed window. Values update when the training code writes evidence and can repeat between heartbeats. |
| `cpu` | System CPU utilization observed by the notebook over the sampling interval. The initial value is null. |
| `gpus` | Each visible board's index, utilization percentage, used MiB and total MiB, sampled with NVIDIA's tool. An empty array means telemetry was unavailable. |
| `comment`, `status` | Short description of preparation, a local checkpoint, completion or failure. Closing normally sends an immediate final update. |
| `started_at`, `sampled_at`, `seq` | Worker session start, sample time and increasing sequence. These prevent delayed samples from replacing newer data. |

## Storage and history

Four workers every 30 seconds generate **11,520 heartbeat requests per 24 hours** of continuous operation. The latest-status table has at most four rows. History records observed run, epoch, session and status transitions, plus one summary after 15 minutes without a recorded event. It keeps the most recent 40 events **per worker**, for a maximum of 160 event rows. The dashboard displays the latest 40 across the four. Very short transitions between samples may only appear in the Kaggle evidence.

This count-based cap works while workers are offline and needs no scheduled cleanup. If you prefer a strict time window as well, use **Integrations → Cron** in Supabase and schedule `delete from public.kd_events where recorded_at < now() - interval '5 hours';` every 15 minutes. Change 5 to 2 for a shorter window. This bounds application rows; request counts, Realtime messages, indexes and database logs have separate usage. Use the Supabase Usage page to see actual consumption. Keep long-term experiment metrics, predictions and recovery checkpoints in the existing Hugging Face repository. [Supabase Cron](https://supabase.com/docs/guides/cron/quickstart)
