import inspect

from api import main
from api import database


def test_api_endpoints_are_async():
    assert inspect.iscoroutinefunction(main.audit)
    assert inspect.iscoroutinefunction(main.results)
    assert inspect.iscoroutinefunction(main.benchmark)


def test_database_helpers_are_async_and_use_aiosqlite_module():
    assert inspect.iscoroutinefunction(database.init_db)
    assert inspect.iscoroutinefunction(database.insert_audit_run)
    assert inspect.iscoroutinefunction(database.get_audit_run)
    assert "sqlite3" not in database.__dict__
