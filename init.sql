CREATE TABLE IF NOT EXISTS forecast (
  timestamp TIMESTAMP WITHOUT TIME ZONE PRIMARY KEY,
  demand NUMERIC,
  price NUMERIC,
  solar NUMERIC
);

CREATE TABLE IF NOT EXISTS optimization_results (
  timestamp TIMESTAMP WITHOUT TIME ZONE PRIMARY KEY,
  charge NUMERIC,
  discharge NUMERIC,
  soc NUMERIC,
  grid NUMERIC,
  solar_gen NUMERIC
);

COPY forecast
FROM '/data/industrial/combined.csv'
WITH CSV HEADER;
