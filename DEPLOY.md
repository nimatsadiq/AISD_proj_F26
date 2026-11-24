# Deploying the prediction service

What gets deployed is `serve/`, the thin HTTP layer that reads the precomputed
prediction store. The model is not in the image. Training runs separately, on a
schedule, in `.github/workflows/monthly-warnings.yml`.

That split is the design decision, not an accident: the serving container has
no scikit-learn, no pandas and no training data, so it is small, it starts in
under a second, and a dependency change in the modelling stack cannot break the
live endpoint.

## Secrets

There are none. The service reads one local JSON file and makes no outbound
calls, so there is no key, token or connection string anywhere in this repo or
in the image. Nothing needs to be injected at runtime.

The only credential in the whole project is the AWS key pair used to reach the
EC2 instance, which lives on your laptop as a `.pem` file and is never
committed. `.gitignore` excludes `*.pem`.

## 1. Build and test locally first

Debug on the laptop, where it is free.

```bash
docker build -t spike-watch .
docker run -d -p 8000:8000 --name spike spike-watch

curl http://localhost:8000/health
curl "http://localhost:8000/warnings?k=5"
open http://localhost:8000/docs          # interactive schema

python -m serve.bench --out bench_local.json

docker stop spike && docker rm spike
```

If `/health` reports `"stale": true`, the prediction store is behind. Refresh it
before deploying:

```bash
python -m src.make_warnings
```

## 2. Deploy to AWS EC2

This follows the course cloud-credits guide, Part C step 3B.

**Launch the instance.** EC2 → Launch instance. Amazon Linux 2023,
`t3.micro` (Free Tier eligible). Create a key pair and download the `.pem`.
Under Network settings allow **SSH (22) from My IP** and **HTTP (80) from
anywhere**. Launch, then copy the Public IPv4 address.

**Install Docker.**

```bash
chmod 400 my-key.pem
ssh -i my-key.pem ec2-user@<PUBLIC_IP>

sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
exit                                   # log out and back in for the group
```

**Build and run.**

```bash
ssh -i my-key.pem ec2-user@<PUBLIC_IP>

git clone https://github.com/nimatsadiq/AISD_proj_F26.git
cd AISD_proj_F26
docker build -t spike-watch .
docker run -d --restart unless-stopped -p 80:8000 --name spike spike-watch

curl http://localhost/health
```

**Check it from outside.** In a browser: `http://<PUBLIC_IP>/docs`

That URL is the deliverable. Put it on the evaluation form.

**Benchmark the deployed endpoint.** From your laptop:

```bash
python -m serve.bench --url http://<PUBLIC_IP> --n 200 --out bench_cloud.json
```

Commit both `bench_local.json` and `bench_cloud.json`. The gap between them is
network round trip, not compute, and being able to say that is the point of
having both.

**Redeploying after a push:**

```bash
cd AISD_proj_F26 && git pull
docker build -t spike-watch . && docker stop spike && docker rm spike
docker run -d --restart unless-stopped -p 80:8000 --name spike spike-watch
```

## 3. Cost

| Item | Rate | Monthly |
|---|---|---|
| EC2 `t3.micro`, 24/7 | Free Tier 750 h/month for 12 months | $0 |
| after Free Tier | ~$0.0104/h in `us-east-1` | ~$7.60 |
| EBS 8 GB gp3 | ~$0.08/GB-month | ~$0.64 |
| Data out | first 100 GB/month free | ~$0 at this volume |
| GitHub Actions | 2,000 min/month free, this job uses ~5 | $0 |
| **Total while on Free Tier** | | **$0** |
| **Total after** | | **~$8.25/month** |

Worth saying out loud: the whole system costs under ten dollars a month to run,
which matters because the organisation that would use it is a humanitarian
agency, not a tech company.

**Azure alternative,** if the AWS card verification is a problem. Azure for
Students is $100 with no credit card, and the deploy is one command:

```bash
az login
az containerapp up --name spike-watch --resource-group rg-aisd-spike \
  --location westeurope --source . --ingress external --target-port 8000
```

## 4. Before the evaluation

- [ ] `/health` returns `"status": "ok"`, not `"degraded"`
- [ ] `/docs` loads in a browser from a machine that is not yours
- [ ] `bench_cloud.json` committed, with p50 and p95 you can quote
- [ ] `git grep -i "aws_secret\|api_key\|password"` returns nothing
- [ ] `*.pem` is in `.gitignore` and no key is in the repo
- [ ] the instance has `--restart unless-stopped`, so a reboot does not kill the demo
