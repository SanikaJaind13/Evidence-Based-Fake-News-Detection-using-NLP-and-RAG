"""Phase 0: schema migration, strict per-user scoping, constraints, cascades."""
import os
import re
import sqlite3
import tempfile
import threading
import unittest
from unittest import mock

from src import database

TS_FORMAT = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

# The ORIGINAL schema, copied from the pre-Phase-0 code, to build a genuine legacy DB.
LEGACY_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    input_text TEXT NOT NULL,
    overall_verdict TEXT NOT NULL,
    overall_confidence REAL NOT NULL,
    summary TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_created ON history (created_at DESC);
"""


def sample(verdict="Supported", text="Some claim text", **extra):
    d = {"created_at": "2026-11-01 10:00:00 UTC", "input_text": text,
         "overall_verdict": verdict, "overall_confidence": 0.8,
         "summary": "1 claim(s) checked.", "claim_results": []}
    d.update(extra)
    return d


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "t.db")

    def tearDown(self):
        self.tmp.cleanup()

    def raw(self):
        conn = sqlite3.connect(self.db)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def add_user(self, email):
        database.init_db(self.db)
        with self.raw() as conn:
            return conn.execute("INSERT INTO users (email, password_hash) VALUES (?, 'x')",
                                (email,)).lastrowid

    def count(self, table, where="1=1", args=()):
        with self.raw() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", args).fetchone()[0]

    def tables(self):
        with self.raw() as conn:
            return {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}


class TestMigration(Base):
    def test_fresh_database(self):
        database.init_db(self.db)
        with self.raw() as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0],
                             database.SCHEMA_VERSION)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(history)")}
        self.assertTrue({"user_id", "input_type", "language", "sentiment", "category"} <= cols)
        self.assertTrue({"history", "users", "sessions", "summaries", "user_interests",
                         "conversations", "messages", "news_articles", "article_hits",
                         "activity_log"} <= self.tables())

    def _make_legacy(self, n=3):
        conn = sqlite3.connect(self.db)
        conn.executescript(LEGACY_SCHEMA)
        for i in range(n):
            conn.execute(
                "INSERT INTO history (created_at, input_text, overall_verdict, "
                "overall_confidence, summary, result_json) VALUES (?,?,?,?,?,?)",
                (f"2026-10-0{i + 1} 09:00:00 UTC", f"भारत legacy claim {i} ✔",
                 "Supported", 0.7, "s", '{"overall_verdict": "Supported", "claim_results": []}'))
        conn.commit()
        conn.close()

    def test_legacy_rows_survive_and_stay_unowned(self):
        self._make_legacy(3)
        database.init_db(self.db)
        with self.raw() as conn:
            rows = conn.execute("SELECT input_text, user_id, input_type, language "
                                "FROM history ORDER BY id").fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][0], "भारत legacy claim 0 ✔")
        self.assertTrue(all(r[1] is None and r[2] is None and r[3] is None for r in rows))

    def test_legacy_rows_not_reassigned_when_users_register(self):
        self._make_legacy(2)
        uid = self.add_user("first@example.com")  # first registered user
        self.assertEqual(self.count("history", "user_id IS NULL"), 2)
        self.assertEqual(database.list_history(self.db, user_id=uid), [])
        self.assertEqual(len(database.list_history(self.db)), 2)  # still unowned

    def test_old_api_still_reads_migrated_legacy_rows(self):
        self._make_legacy(2)
        rows = database.list_history(self.db)  # exactly how app.py calls it today
        self.assertEqual(len(rows), 2)
        data = database.get_analysis(self.db, rows[0]["id"])
        self.assertEqual(data["overall_verdict"], "Supported")

    def test_idempotent(self):
        self._make_legacy(2)
        database.init_db(self.db)
        database.init_db(self.db)
        self.assertEqual(self.count("history"), 2)
        with self.raw() as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(history)")]
        self.assertEqual(len(cols), len(set(cols)))

    def test_failed_migration_rolls_back_completely(self):
        self._make_legacy(2)

        def boom(conn):
            conn.execute("CREATE TABLE half_done (x)")
            raise RuntimeError("simulated crash")

        with mock.patch.object(database, "_migrate_to_v2", boom):
            with self.assertRaises(RuntimeError):
                database.init_db(self.db)
        with self.raw() as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 0)
        self.assertNotIn("half_done", self.tables())
        self.assertNotIn("users", self.tables())
        self.assertEqual(self.count("history"), 2)
        database.init_db(self.db)  # a later, healthy start upgrades fine
        self.assertIn("users", self.tables())
        self.assertEqual(self.count("history"), 2)

    def test_concurrent_init(self):
        errors = []

        def work():
            try:
                database.init_db(self.db)
            except Exception as exc:  # pragma: no cover - only on failure
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        with self.raw() as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0],
                             database.SCHEMA_VERSION)


class TestScoping(Base):
    def test_users_are_isolated_from_each_other_and_from_legacy(self):
        a, b = self.add_user("a@x.com"), self.add_user("b@x.com")
        legacy = database.save_analysis(self.db, sample(text="legacy"))
        ida = database.save_analysis(self.db, sample(text="A's claim"), user_id=a)
        idb = database.save_analysis(self.db, sample("Contradicted", "B's claim"), user_id=b)

        self.assertEqual([r["id"] for r in database.list_history(self.db, user_id=a)], [ida])
        self.assertEqual([r["id"] for r in database.list_history(self.db, user_id=b)], [idb])
        self.assertEqual([r["id"] for r in database.list_history(self.db)], [legacy])

        self.assertIsNone(database.get_analysis(self.db, ida, user_id=b))   # B cannot read A
        self.assertIsNone(database.get_analysis(self.db, ida))              # nor can "no user"
        self.assertIsNone(database.get_analysis(self.db, legacy, user_id=a))
        self.assertIsNotNone(database.get_analysis(self.db, ida, user_id=a))

    def test_delete_and_clear_cannot_cross_users(self):
        a, b = self.add_user("a@x.com"), self.add_user("b@x.com")
        legacy = database.save_analysis(self.db, sample())
        ida = database.save_analysis(self.db, sample(), user_id=a)
        database.save_analysis(self.db, sample(), user_id=b)
        database.save_analysis(self.db, sample(), user_id=b)

        self.assertFalse(database.delete_analysis(self.db, ida, user_id=b))
        self.assertFalse(database.delete_analysis(self.db, ida))
        self.assertFalse(database.delete_analysis(self.db, legacy, user_id=a))
        self.assertEqual(self.count("history"), 4)

        self.assertEqual(database.clear_history(self.db, user_id=b), 2)
        self.assertEqual(self.count("history"), 2)  # A's row and the legacy row remain
        self.assertEqual(database.clear_history(self.db), 1)  # only the legacy row
        self.assertEqual(database.list_history(self.db, user_id=a)[0]["id"], ida)

    def test_metadata_columns_saved_and_listed(self):
        uid = self.add_user("a@x.com")
        database.save_analysis(
            self.db, sample(input_type="url", language="hi", sentiment="negative",
                            category="Politics"), user_id=uid)
        row = database.list_history(self.db, user_id=uid)[0]
        self.assertEqual((row["input_type"], row["language"], row["sentiment"], row["category"]),
                         ("url", "hi", "negative", "Politics"))

    def test_metadata_defaults_to_null(self):
        database.save_analysis(self.db, sample())
        row = database.list_history(self.db)[0]
        self.assertIsNone(row["input_type"])
        self.assertIsNone(row["language"])

    def test_unknown_user_rejected_by_foreign_key(self):
        database.init_db(self.db)
        with self.assertRaises(sqlite3.IntegrityError):
            database.save_analysis(self.db, sample(), user_id=999)

    def test_bad_user_id_types_rejected(self):
        for bad in ("1", True, 1.5):
            with self.assertRaises(ValueError):
                database.list_history(self.db, user_id=bad)


class TestConstraintsAndCascades(Base):
    def test_email_unique_case_insensitive(self):
        self.add_user("Person@Example.com")
        with self.assertRaises(sqlite3.IntegrityError):
            self.add_user("person@example.COM")

    def test_new_table_timestamps_are_utc_text(self):
        uid = self.add_user("a@x.com")
        with self.raw() as conn:
            created = conn.execute("SELECT created_at FROM users WHERE id=?", (uid,)).fetchone()[0]
        self.assertRegex(created, TS_FORMAT)

    def test_message_role_and_interest_constraints(self):
        uid = self.add_user("a@x.com")
        with self.raw() as conn:
            cid = conn.execute("INSERT INTO conversations (user_id) VALUES (?)", (uid,)).lastrowid
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO messages (conversation_id, role, content) "
                             "VALUES (?, 'system', 'x')", (cid,))
            conn.execute("INSERT INTO user_interests (user_id, kind, value) VALUES (?, 'custom', 'ISRO')", (uid,))
            with self.assertRaises(sqlite3.IntegrityError):  # duplicate, case-insensitive
                conn.execute("INSERT INTO user_interests (user_id, kind, value) VALUES (?, 'custom', 'isro')", (uid,))
            with self.assertRaises(sqlite3.IntegrityError):  # bad kind
                conn.execute("INSERT INTO user_interests (user_id, kind, value) VALUES (?, 'other', 'x')", (uid,))

    def test_deleting_a_user_removes_private_data_but_keeps_legacy_and_anonymises_activity(self):
        uid, other = self.add_user("a@x.com"), self.add_user("b@x.com")
        legacy = database.save_analysis(self.db, sample(text="legacy"))
        database.save_analysis(self.db, sample(), user_id=uid)
        database.save_analysis(self.db, sample(), user_id=other)
        with self.raw() as conn:
            conn.execute("INSERT INTO sessions (user_id, token_hash, expires_at) VALUES (?, 'h', '2030-01-01 00:00:00')", (uid,))
            conn.execute("INSERT INTO summaries (user_id, source_type, short_summary, detailed_summary) VALUES (?, 'text', 's', 'd')", (uid,))
            conn.execute("INSERT INTO user_interests (user_id, kind, value) VALUES (?, 'category', 'Health')", (uid,))
            cid = conn.execute("INSERT INTO conversations (user_id) VALUES (?)", (uid,)).lastrowid
            conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, 'user', 'hi')", (cid,))
            conn.execute("INSERT INTO activity_log (user_id, feature) VALUES (?, 'check')", (uid,))
            aid = conn.execute("INSERT INTO news_articles (url, title) VALUES ('https://n.example/1', 'T')").lastrowid
            conn.execute("INSERT INTO article_hits (article_id, user_id, origin) VALUES (?, ?, 'news')", (aid, uid))
            conn.execute("DELETE FROM users WHERE id=?", (uid,))
        for table in ("sessions", "summaries", "user_interests", "conversations", "messages"):
            self.assertEqual(self.count(table), 0, table)
        self.assertEqual(self.count("history", "user_id=?", (uid,)), 0)
        self.assertEqual(self.count("history", "user_id=?", (other,)), 1)   # others untouched
        self.assertIsNotNone(database.get_analysis(self.db, legacy))        # legacy untouched
        self.assertEqual(self.count("activity_log", "user_id IS NULL"), 1)  # kept, anonymised
        self.assertEqual(self.count("article_hits", "user_id IS NULL"), 1)

    def test_article_url_unique_and_hits_cascade(self):
        database.init_db(self.db)
        with self.raw() as conn:
            aid = conn.execute("INSERT INTO news_articles (url, title) VALUES ('https://n.example/1', 'T')").lastrowid
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO news_articles (url, title) VALUES ('https://n.example/1', 'T2')")
            conn.execute("INSERT INTO article_hits (article_id, origin) VALUES (?, 'news')", (aid,))
            conn.execute("DELETE FROM news_articles WHERE id=?", (aid,))
        self.assertEqual(self.count("article_hits"), 0)


if __name__ == "__main__":
    unittest.main()