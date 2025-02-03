CREATE TABLE IF NOT EXISTS  example_schedule (
  timestamp VARCHAR PRIMARY KEY,
  demand_scheduled VARCHAR,
  solar VARCHAR,
  price VARCHAR,
  price2 VARCHAR,
  demand_bldg VARCHAR
);

COPY example_schedule
FROM '/data/combined.csv'
WITH CSV HEADER;