import psycopg2
from flask import current_app

from ferelight.models import Multimediaobject


def objectinfo_objectid_get(objectid):  # noqa: E501
    """Get the information of an object.

     # noqa: E501

    :param objectid: The unique identifier of the object.
    :type objectid: str

    :rtype: Union[Multimediaobject, Tuple[Multimediaobject, int], Tuple[Multimediaobject, int, Dict[str, str]]
    """
    with psycopg2.connect(dbname=current_app.config['DBNAME'], user=current_app.config['DBUSER'],
                          password=current_app.config['DBPASSWORD'], host=current_app.config['DBHOST'],
                          port=current_app.config['DBPORT']) as conn:
        cur = conn.cursor()
        cur.execute(f"""SELECT objectid, mediatype, name, path FROM cineast_multimediaobject WHERE objectid = %s""",
                    (objectid,))
        (objectid, mediatype, name, path) = cur.fetchone()
        return Multimediaobject(objectid=objectid, mediatype=mediatype, name=name, path=path)
