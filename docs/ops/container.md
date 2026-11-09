# Container image

The root `Dockerfile` builds the `caliche_server` binary into an image that sets `CALICHE_ENV=production`. In that profile, the process refuses to start until `JWT_ENABLED`, `CALICHE_REQUIRE_AUTH`, `JWT_AUDIENCE`, and `JWT_ISSUER` are set, and it also requires `PLUGIN_HTTP_ALLOWLIST`. The full list of settings is in the [production authentication checklist](production-auth-checklist.md).

Inside the container, the server listens on `0.0.0.0` so that Docker can publish the port. Publish the port on the loopback interface of the host unless the host is meant to be reachable from other machines:

```bash
docker build -t caliche-server .
docker run --rm -p 127.0.0.1:3000:3000 \
  -e JWT_ENABLED=true \
  -e JWT_PUBLIC_KEY="<pem>" \
  -e JWT_AUDIENCE="<audience>" \
  -e JWT_ISSUER="<issuer>" \
  -e CALICHE_REQUIRE_AUTH=true \
  -e PLUGIN_HTTP_ALLOWLIST="<hosts>" \
  caliche-server
```

Setting `CALICHE_ALLOW_INSECURE_PRODUCTION=1` starts the image without these authentication settings. Use it only for a local test.

`docker-compose.yml` publishes the ports of its development services on `127.0.0.1` only. The development passwords for those services are in `.env.example`, under the `STACK_*` variables.
