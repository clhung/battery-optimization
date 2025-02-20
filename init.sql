CREATE TABLE IF NOT EXISTS example_schedule (
  demand NUMERIC,
  solar NUMERIC,
  price NUMERIC,
  timestamp TIMESTAMP WITHOUT TIME ZONE
);

COPY example_schedule
FROM '/data/combined.csv'
WITH CSV HEADER;
