import os
import secrets
import sqlite3
from pathlib import Path

from flask import Flask, abort, flash, g, redirect, render_template, request, session, url_for


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(SECRET_KEY=os.environ.get('SECRET_KEY') or secrets.token_hex(32),
                            DATABASE=str(Path(app.instance_path) / 'shopping.sqlite'),
                            MAX_CONTENT_LENGTH=128 * 1024,
                            SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')
    if test_config:
        app.config.update(test_config)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    def db():
        if 'db' not in g:
            g.db = sqlite3.connect(app.config['DATABASE'])
            g.db.row_factory = sqlite3.Row
            g.db.execute('PRAGMA foreign_keys = ON')
        return g.db

    @app.teardown_appcontext
    def close_db(error=None):
        connection = g.pop('db', None)
        if connection:
            connection.close()

    with app.app_context():
        db().executescript('''
            CREATE TABLE IF NOT EXISTS lists (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY, list_id INTEGER NOT NULL REFERENCES lists(id) ON DELETE CASCADE,
                name TEXT NOT NULL, quantity TEXT NOT NULL DEFAULT '1',
                checked INTEGER NOT NULL DEFAULT 0);
        ''')

    @app.before_request
    def csrf():
        if 'csrf' not in session:
            session['csrf'] = secrets.token_hex(32)
        if request.method == 'POST' and not secrets.compare_digest(session['csrf'], request.form.get('csrf', '')):
            abort(400, 'This form has expired. Refresh the page and try again.')

    @app.get('/')
    @app.get('/lists/<int:list_id>')
    def index(list_id=None):
        lists = db().execute('''SELECT l.*, COUNT(i.id) AS total, COALESCE(SUM(i.checked),0) AS done
            FROM lists l LEFT JOIN items i ON l.id=i.list_id GROUP BY l.id
            ORDER BY l.updated_at DESC, l.id DESC''').fetchall()
        current = None
        items = []
        if list_id is not None:
            current = db().execute('SELECT * FROM lists WHERE id=?', (list_id,)).fetchone()
            if current is None:
                abort(404)
            items = db().execute('SELECT * FROM items WHERE list_id=? ORDER BY id', (list_id,)).fetchall()
        return render_template('index.html', lists=lists, current=current, items=items)

    @app.post('/lists/save')
    def save():
        name = request.form.get('name', '').strip()
        names = request.form.getlist('item_name')
        quantities = request.form.getlist('quantity')
        checked = set(request.form.getlist('checked'))
        if not name or len(name) > 80 or len(names) > 200 or len(names) != len(quantities):
            abort(400, 'Use a list name of 1–80 characters and at most 200 items.')
        rows = []
        for i, (item, quantity) in enumerate(zip(names, quantities)):
            item, quantity = item.strip(), quantity.strip() or '1'
            if len(item) > 120 or len(quantity) > 30:
                abort(400, 'Item names can have up to 120 characters; quantities up to 30.')
            if item:
                rows.append((item, quantity, int(str(i) in checked)))
        list_id = request.form.get('list_id', type=int)
        connection = db()
        with connection:
            if list_id is not None:
                if not connection.execute('SELECT id FROM lists WHERE id=?', (list_id,)).fetchone():
                    abort(404)
                connection.execute('UPDATE lists SET name=?, updated_at=CURRENT_TIMESTAMP WHERE id=?', (name, list_id))
                connection.execute('DELETE FROM items WHERE list_id=?', (list_id,))
            else:
                list_id = connection.execute('INSERT INTO lists(name) VALUES(?)', (name,)).lastrowid
            connection.executemany('INSERT INTO items(list_id,name,quantity,checked) VALUES(?,?,?,?)',
                                   [(list_id, *row) for row in rows])
        flash('Shopping list saved.')
        return redirect(url_for('index', list_id=list_id))

    @app.post('/lists/<int:list_id>/delete')
    def delete(list_id):
        with db():
            result = db().execute('DELETE FROM lists WHERE id=?', (list_id,))
            if not result.rowcount:
                abort(404)
        flash('Shopping list deleted.')
        return redirect(url_for('index'))

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(413)
    def error(error):
        return render_template('error.html', error=error), error.code

    return app


app = create_app()
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
