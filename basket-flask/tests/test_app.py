import sqlite3
import tempfile
import unittest
from pathlib import Path

from werkzeug.datastructures import MultiDict
from app import create_app


class ShoppingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.path = str(Path(self.temp.name) / 'test.sqlite')
        self.app = create_app({'TESTING': True, 'DATABASE': self.path, 'SECRET_KEY': 'test'})
        self.client = self.app.test_client()
        self.client.get('/')
        with self.client.session_transaction() as session:
            self.token = session['csrf']

    def tearDown(self):
        self.temp.cleanup()

    def save(self, fields):
        return self.client.post('/lists/save', data=MultiDict([('csrf', self.token), *fields]))

    def test_create_update_persist_delete(self):
        response = self.save([('name', 'Weekly shop'), ('item_name', 'Milk'), ('quantity', '2 pints'),
                              ('item_name', 'Bread'), ('quantity', '1'), ('checked', '1')])
        self.assertEqual(response.status_code, 302)
        self.assertIn(b'Milk', self.client.get('/lists/1').data)
        self.assertEqual(self.save([('list_id', '1'), ('name', 'Weekend'), ('item_name', 'Apples'),
                                   ('quantity', '6'), ('checked', '0')]).status_code, 302)
        other_app = create_app({'TESTING': True, 'DATABASE': self.path, 'SECRET_KEY': 'test'})
        page = other_app.test_client().get('/lists/1').data
        self.assertIn(b'Apples', page)
        self.assertNotIn(b'value="Milk"', page)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT name,quantity,checked FROM items').fetchall(), [('Apples', '6', 1)])
        db.close()
        self.assertEqual(self.client.post('/lists/1/delete', data={'csrf': self.token}).status_code, 302)
        self.assertEqual(self.client.get('/lists/1').status_code, 404)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM items').fetchone()[0], 0)
        db.close()

    def test_validation_and_csrf(self):
        self.assertEqual(self.save([('name', '  ')]).status_code, 400)
        self.assertEqual(self.save([('name', 'List'), ('item_name', 'Milk')]).status_code, 400)
        self.assertEqual(self.client.post('/lists/save', data={'name': 'No token'}).status_code, 400)
        self.assertEqual(self.save([('list_id', '999'), ('name', 'Missing')]).status_code, 404)

    def test_empty_list_and_escaping(self):
        self.assertEqual(self.save([('name', '<script>alert(1)</script>')]).status_code, 302)
        page = self.client.get('/lists/1').data
        self.assertIn(b'&lt;script&gt;', page)
        self.assertNotIn(b'<script>alert(1)</script>', page)


if __name__ == '__main__':
    unittest.main()
