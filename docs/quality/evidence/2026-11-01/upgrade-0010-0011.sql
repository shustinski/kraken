BEGIN;

-- Running upgrade 20261025_0010 -> 20261029_0011

ALTER TABLE background_jobs ADD COLUMN execution_state JSON;

UPDATE alembic_version SET version_num='20261029_0011' WHERE alembic_version.version_num = '20261025_0010';

COMMIT;

