CREATE TABLE IF NOT EXISTS  example_schedule (
  timestamp VARCHAR PRIMARY KEY,
  demand_scheduled NUMERIC,
  solar NUMERIC,
  price NUMERIC,
  price2 NUMERIC,
  demand_bldg NUMERIC
);

COPY example_schedule
FROM '/data/combined.csv'
WITH CSV HEADER;
