from fastapi.testclient import TestClient

import src.server as server
from src.server import app, sanitize, is_book, re_book

model, embeddings, database = engines = server._get_engines()

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_sanitize():
    san1 = sanitize("ThiS? Is,.:;'\" text -_  >< |lalA\n What!!")
    print("sanitized = ", san1)
    assert san1 == "this is,.:; text -_ lala what"
    
def test_re_book():
    m1 = re_book.match("genesys")
    assert m1.groups() == (None, None, 'genesys', None, None, None)
    m1a = re_book.match("6   genesys")
    assert m1a.groups() == ('6   ', '6', 'genesys', None, None, None)
    ## BUGS here chapter and verse get reverted but I haven't managed to fix them
    m2 = re_book.match("genesys 1")
    assert m2.groups() == (None, None, 'genesys', ' 1', None, '1')
    m2a = re_book.match("5 genesys 1")
    assert m2a.groups() == ('5 ', '5', 'genesys', ' 1', None, '1')
    ## END BUGS
    m3 = re_book.match("genesys 1:2")
    assert m3.groups() == (None, None, 'genesys', ' 1:2', ' 1', '2') 
    m3a = re_book.match("4 genesys 1:2")
    assert m3a.groups() == ('4 ', '4', 'genesys', ' 1:2', ' 1', '2')
    m3b = re_book.match("4: genesys 1:2")
    assert m3b.groups() == ('4: ', '4', 'genesys', ' 1:2', ' 1', '2')
    
def test_is_book():
    m1 = is_book("genesis", database['book'])
    assert m1 == 0
    m1a = is_book("6   genesis", database['book'])
    assert m1a == None
    m2 = is_book("genesis 2", database['book'])
    assert m2 == 31
    m2a = is_book("5 genesis 1", database['book'])
    assert m2a == None
    m3 = is_book("genesis 3:2", database['book'])
    assert m3 == 57
    m3a = is_book("1 samuel 1:2", database['book'])
    assert m3a == 7214
    m3b = is_book("2: timothy 3:6", database['book'])
    assert m3b == 29859
    