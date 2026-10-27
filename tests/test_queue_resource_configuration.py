import os
from unittest.mock import patch

import pytest

from kraken_manager.domain.pipeline import PipelineResources
from kraken_server.agent_resources import QueueResourceBudget
from kraken_server.configuration import ServerConfig,write_config


def config(tmp_path,**arguments):
    return write_config(tmp_path/'server.toml',database_url='postgresql+psycopg://unused:unused@127.0.0.1/unused',
        blob_root=tmp_path/'blobs',source_root=tmp_path/'source',derived_root=tmp_path/'derived',**arguments)


def test_explicit_budget_roundtrips_file_and_all_environment_dimensions(tmp_path):
    budget=QueueResourceBudget(3,PipelineResources(cpu=4,gpu=1,memory_mb=4096))
    written=config(tmp_path,queue_budget=budget)
    loaded=ServerConfig.load(written.path)
    assert loaded.queue_budget==budget
    with patch.dict(os.environ,{},clear=False):
        loaded.apply_to_environment()
        assert QueueResourceBudget.from_environment()==budget


@pytest.mark.parametrize('table', ['cpu=2', 'max_active_jobs=1\ncpu=2\ngpu=true\nmemory_mb=100',
    'max_active_jobs=1\ncpu=2\ngpu=1\nmemory_mb=100\nunknown=2'], ids=['partial','boolean','unknown'])
def test_invalid_budget_table_fails_before_server_start(tmp_path,table):
    value=config(tmp_path)
    with value.path.open('a',encoding='utf-8') as writer:
        writer.write('\n[compute_admission]\n'+table+'\n')
    with pytest.raises(ValueError):ServerConfig.load(value.path)


def test_absent_budget_remains_explicitly_unconfigured(tmp_path):
    assert config(tmp_path).queue_budget is None


def test_invalid_budget_argument_does_not_write_a_database_secret(tmp_path):
    with pytest.raises(ValueError,match='validated'):
        config(tmp_path,queue_budget={'cpu':1})
    assert not (tmp_path/'database-url.secret').exists()
