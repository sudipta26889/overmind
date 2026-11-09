# Production Console for the self-hosted stack. Build context is frontend/.
FROM oven/bun:1.3.5 AS build
WORKDIR /app
COPY package.json bun.lock ./
COPY patches ./patches
RUN bun install --frozen-lockfile
COPY . .
# Vite bakes these into the bundle. A process env value beats frontend/.env.production.
ARG VITE_API_BASE_URL
# frontend/.env.production turns MCP off; the self-hosted Console keeps it on.
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL VITE_SELF_HOSTED=true VITE_CLERK_PUBLISHABLE_KEY= VITE_FEATURE_MCP=true
RUN bun run build

FROM nginx:stable-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY <<'NGINX' /etc/nginx/conf.d/default.conf
server {
    listen 80;
    root /usr/share/nginx/html;

    location /assets/ {
        add_header Cache-Control "public, max-age=31536000, immutable";
        try_files $uri =404;
    }

    location / {
        add_header Cache-Control "no-cache";
        try_files $uri /index.html;
    }
}
NGINX
