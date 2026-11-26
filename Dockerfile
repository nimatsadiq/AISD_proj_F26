# Spike Watch prediction service.
#
# Serves the shortlist that src/make_warnings.py computed earlier. The model is
# NOT in this image: training happens in the scheduled batch job, which is why
# this stays small and starts in under a second.
#
# Local:
#   docker build -t spike-watch .
#   docker run -d -p 8000:8000 --name spike spike-watch
#   curl http://localhost:8000/warnings
#
# AWS EC2: see DEPLOY.md

FROM python:3.12-slim

# Run as a non-root user. Nothing here needs privileges, and a container that
# cannot write to its own filesystem cannot be made to persist anything.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# Dependencies first so code edits do not reinstall them on rebuild.
COPY serve/requirements.txt ./serve/requirements.txt
RUN pip install --no-cache-dir -r serve/requirements.txt

COPY serve/ ./serve/
# The batch job's output, committed to the repo so the image is self-contained
# and needs no database.
COPY data/processed/dashboard_warnings.json ./data/processed/dashboard_warnings.json

# The one page a person looks at, served from the same origin as the data it
# reads. A browser will not let an HTTPS page call an HTTP service, and this
# service has no certificate, so a dashboard hosted anywhere else could not
# read it. Same origin, relative requests, no such problem. It is one static
# file and adds nothing measurable to the image.
COPY dashboard.html ./dashboard.html

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

ENV PORT=8000
CMD ["sh", "-c", "uvicorn serve.main:app --host 0.0.0.0 --port ${PORT}"]
