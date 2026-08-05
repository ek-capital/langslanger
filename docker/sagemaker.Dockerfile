FROM ghcr.io/ek-capital/langslanger:nightly

COPY serve /usr/bin/serve
RUN chmod 777 /usr/bin/serve

ENTRYPOINT [ "/usr/bin/serve" ]
