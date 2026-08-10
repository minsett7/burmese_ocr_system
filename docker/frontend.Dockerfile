FROM node:20-alpine AS build
WORKDIR /frontend
COPY services/insurance-claim-ui/frontend/package.json services/insurance-claim-ui/frontend/package-lock.json ./
RUN npm ci
COPY services/insurance-claim-ui/frontend/ ./
ARG VITE_API_BASE_URL=.
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

FROM nginx:1.27-alpine
COPY docker/frontend-nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /frontend/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=5 \
  CMD wget -qO- http://127.0.0.1/healthz >/dev/null || exit 1
