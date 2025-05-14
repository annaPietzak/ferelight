import numpy as np
import psycopg2
import torch
from flask import current_app
from pgvector.psycopg2 import register_vector

from ferelight.controllers import tokenizer, model
from ferelight.models import Scoredsegment
from ferelight.models.multimediaobject import Multimediaobject  # noqa: E501
from ferelight.models.multimediasegment import Multimediasegment  # noqa: E501
from ferelight.models.objectinfos_post_request import ObjectinfosPostRequest  # noqa: E501
from ferelight.models.query_post_request import QueryPostRequest  # noqa: E501
from ferelight.models.scoredsegment import Scoredsegment  # noqa: E501
from ferelight.models.segmentbytime_post200_response import SegmentbytimePost200Response  # noqa: E501
from ferelight.models.segmentinfos_post_request import SegmentinfosPostRequest  # noqa: E501
from ferelight import util


def get_connection(database):
    return psycopg2.connect(dbname=database, user=current_app.config['DBUSER'],
                            password=current_app.config['DBPASSWORD'], host=current_app.config['DBHOST'],
                            port=current_app.config['DBPORT'])


def objectinfo_database_objectid_get(database, objectid):  # noqa: E501
    """Get the information of an object.

     # noqa: E501

    :param database: The name of the database to query for the object.
    :type database: str
    :param objectid: The unique identifier of the object.
    :type objectid: str

    :rtype: Union[Multimediaobject, Tuple[Multimediaobject, int], Tuple[Multimediaobject, int, Dict[str, str]]
    """
    with get_connection(database) as conn:
        cur = conn.cursor()
        cur.execute(f"""SELECT objectid, mediatype, name, path FROM cineast_multimediaobject WHERE objectid = %s""",
                    (objectid,))
        (objectid, mediatype, name, path) = cur.fetchone()
        return Multimediaobject(objectid=objectid, mediatype=mediatype, name=name, path=path)


def objectinfos_post(body):  # noqa: E501
    """Get the information of multiple objects.

     # noqa: E501

    :param objectinfos_post_request:
    :type objectinfos_post_request: dict | bytes

    :rtype: Union[List[Multimediaobject], Tuple[List[Multimediaobject], int], Tuple[List[Multimediaobject], int, Dict[str, str]]
    """
    with get_connection(body['database']) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""SELECT objectid, mediatype, name, path FROM cineast_multimediaobject WHERE objectid = ANY(%s)""",
            (body['objectids'],))
        results = cur.fetchall()

    object_infos = [Multimediaobject(objectid=objectid, mediatype=mediatype, name=name, path=path) for
                    (objectid, mediatype, name, path) in results]

    return object_infos


def objectsegments_database_objectid_get(database, objectid):  # noqa: E501
    """Get the segments of an object.

     # noqa: E501

    :param database: The name of the database to query for the object.
    :type database: str
    :param objectid: The unique identifier of the object.
    :type objectid: str

    :rtype: Union[List[Multimediasegment], Tuple[List[Multimediasegment], int], Tuple[List[Multimediasegment], int, Dict[str, str]]
    """
    with get_connection(database) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
                SELECT segmentid, objectid, segmentnumber, segmentstart, segmentend, segmentstartabs, segmentendabs  
                FROM cineast_segment WHERE objectid = %s""",
            (objectid,))
        results = cur.fetchall()

    segmentinfos = [Multimediasegment(segmentid=segmentid, objectid=objectid, segmentnumber=segmentnumber,
                                      segmentstart=segmentstart, segmentend=segmentend, segmentstartabs=segmentstartabs,
                                      segmentendabs=segmentendabs) for
                    (segmentid, objectid, segmentnumber, segmentstart, segmentend, segmentstartabs, segmentendabs) in
                    results]

    return segmentinfos

###########################################################
def query_post(body):  # noqa: E501
    """Query the FERElight engine.

     # noqa: E501

    :param query_post_request:
    :type query_post_request: dict | bytes

    :rtype: Union[List[Scoredsegment], Tuple[List[Scoredsegment], int], Tuple[List[Scoredsegment], int, Dict[str, str]]
    """
    limit = f'LIMIT {body["limit"]}' if 'limit' in body else ''

    similarity_vector = []
    if 'similaritytext' in body: similarity_vector = vectorize_textinput(body['similaritytext'])

    with get_connection(body['database']) as conn:
        cur = conn.cursor()
        cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(conn)
        # Set index parameter to allow for correct number of results
        if 'limit' in body:
            cur.execute('SET hnsw.ef_search = %s', (body['limit'],))

        if 'ocrtext' in body and not 'similaritytext' in body and not 'asrtext' in body:
            return ocrtext_query(cur, body['ocrtext'], limit)

        elif 'similaritytext' in body and not 'ocrtext' in body and not 'asrtext' in body:
            *inputs, mergeType = [vectorize_textinput(x) for x in body['similaritytext'].split('#') if x != ""]
            
            if mergeType == 'vector_addition':
                input = np.mean(inputs, axis=0)
                result = similaritytext_query(cur, input, limit)
                print("Amount of results " + str(len(result)))
                return result
            
            elif mergeType == 'id_intersection':
                tmp = [[]]
                tmp[0] = similaritytext_query(cur, inputs[0], limit)
                ids = set([x.segmentid for x in tmp[0]])

                if len(inputs) > 1:
                    for input in inputs[1:]:
                        # vector = vectorize_textinput(input)
                        sim = similaritytext_query(cur, input, limit)
                        tmp.append(sim)
                        ids &= set([x.segmentid for x in sim])

                    if len(ids) < 10:
                        length = len(tmp)
                        for i in range(length): 
                            id = set([x.segmentid for x in tmp[i]]) - ids
                            id = f"{id}".replace("{", "").replace("}", "")
                            for j in range(len(inputs)):
                                if i == j: continue
                                cur.execute(
                                    f"""
                                        SELECT id, feature <=> %s AS distance
                                        FROM features_openclip
                                        WHERE id  IN ({id})
                                        ORDER BY distance
                                        {limit}
                                    """,
                                    (inputs[j],)
                                )
                                res = [x for x in evaluate_cursor(cur) if x.score >= 0.17]
                                print("length before " + str(len(tmp[i])) + ", " + str(len(res)))
                                tmp[i] += res
                                print('IDs of ' + str(body['similaritytext'].split("#")[i]) + ' Score of ' + str(body['similaritytext'].split("#")[j]) + ' ' + str(len(tmp[i])))
                            
                tmp = [x for i in range(len(tmp)) for x in tmp[i] ]
                tmp.sort(key=lambda x: x.segmentid)
                result = []
                i = 0
                while i <= (len(tmp) - len(inputs)):
                    avgElement = [tmp[i].segmentid, tmp[i].score, 1]
                    for j in range((i+1), (len(tmp) - 1 )):
                        if tmp[i].segmentid == tmp[j].segmentid:
                            avgElement[1] += tmp[j].score
                            avgElement[2] += 1

                    if avgElement[2] >= len(inputs):
                        result.append(Scoredsegment(avgElement[0], (avgElement[1] / float(avgElement[2]))))
                    i += avgElement[2]

                result.sort(key= lambda x: x.score)
                print("Amount of results " + str(len(result)))
                return result
        
        elif 'asrtext' in body and not 'similaritytext' in body and not 'ocrtext' in body:
            return asrtext_query(cur, body['asrtext'], limit)
        
        elif 'ocrtext' in body and 'similaritytext' in body and not 'asrtext' in body:
            cur.execute(
                f"""
                    SELECT id, feature <=> %s AS distance
                    FROM features_openclip
                    WHERE id IN (
                        SELECT id 
                        FROM features_ocr
                        WHERE feature @@ plainto_tsquery(%s)
                    )
                    ORDER BY distance
                    {limit}
                """,
                (similarity_vector, body['ocrtext'])
            )

        elif 'asrtext' in body and 'similaritytext' in body and 'ocrtext' not in body:
            asr = [x.segmentid for x in asrtext_query(cur, body['asrtext'], limit)]
            ids = f"{asr}".replace("[", "").replace("]", "")
            cur.execute(
                f"""
                    SELECT id, feature <=> %s AS distance
                    FROM features_openclip
                    WHERE id  IN ({ids})
                    ORDER BY distance
                    {limit}
                """,
                (vectorize_textinput(body['similaritytext']),)
            )
            return evaluate_cursor(cur)
        
        elif 'asrtext' in body and 'ocrtext' in body and 'similaritytext' not in body:
            asr = set([x.segmentid for x in asrtext_query(cur, body['asrtext'], limit)])
            ocr = set([x.segmentid for x in ocrtext_query(cur, body['ocrtext'], limit)])
            return [Scoredsegment(segmentid=x, score=1) for x in (asr & ocr)]
        
        elif 'asrtext' in body and 'ocrtext' in body and 'similaritytext' in body:
            asr_ocr = [x.segmentid for x in ocrtext_query(cur, body['ocrtext'], limit) if x in asrtext_query(cur, body['asrtext'], limit)]
            ids =  f"{asr_ocr}".replace("[", "").replace("]", "")

            inputs = body['similaritytext'].split('#')

            tmp = []
            for input in inputs:
                cur.execute(
                    f"""
                        SELECT id, feature <=> %s AS distance
                        FROM features_openclip
                        WHERE id  IN ({ids})
                        ORDER BY distance
                        {limit}
                    """,
                    (vectorize_textinput(input),)
                )   
                tmp += evaluate_cursor(cur)
            
            tmp.sort(key=lambda x: x.segmentid)
            result = []
            i = 0
            while i <= (len(tmp) - len(inputs)):
                avgElement = [tmp[i].segmentid, tmp[i].score, 1]
                for j in range((i+1), (len(tmp) - 1 )):
                    if tmp[i].segmentid == tmp[j].segmentid:
                        avgElement[1] += tmp[j].score
                        avgElement[2] += 1

                if avgElement[2] >= len(inputs):
                    result.append(Scoredsegment(avgElement[0], (avgElement[1] / float(avgElement[2]))))
                i += avgElement[2]
            result.sort(key= lambda x: x.score)
            print("Amount of results " + str(len(result)))
            return result
        
        else:
            return "Not a valid query"
           

# helping functions for query
def vectorize_textinput(input):
    text = tokenizer(input)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()

def similaritytext_query(cur, similarity_vector, limit):
    # Get cosine similarity as score
    cur.execute(
        f"""
            SELECT id, feature <=> %s AS distance
            FROM features_openclip
            ORDER BY distance
            {limit}
        """,
        (similarity_vector,))
    
    return evaluate_cursor(cur)

def ocrtext_query(cur, input, limit):
    cur.execute(
        f"""
            SELECT id, 0 AS distance
            FROM features_ocr WHERE feature @@ plainto_tsquery(%s)
            {limit}
        """,
        (input,))
    
    return evaluate_cursor(cur)

def asrtext_query(cur, input, limit):
    # TODO Implement ASR features
    # return Scoredsegment(segmentid='dummy_id', score=1)
    return ocrtext_query(cur, input, limit)

def evaluate_cursor(cur):
    results = cur.fetchall()
    scored_segments = [Scoredsegment(segmentid=segmentid, score=1 - distance) for (segmentid, distance) in set(results)]
    print("Amount of results", len(scored_segments))
    scored_segments.sort(key= lambda x: x.score)
    return scored_segments

#######################################################

def segmentinfo_database_segmentid_get(database, segmentid):  # noqa: E501
    """Get the information of a segment.

     # noqa: E501

    :param database: The name of the database to query for the segment.
    :type database: str
    :param segmentid: The unique identifier of the segment.
    :type segmentid: str

    :rtype: Union[Multimediasegment, Tuple[Multimediasegment, int], Tuple[Multimediasegment, int, Dict[str, str]]
    """
    with get_connection(database) as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT segmentid, objectid, segmentnumber, segmentstart, segmentend, segmentstartabs, segmentendabs 
            FROM cineast_segment WHERE segmentid = %s
        """,
                    (segmentid,))
        (segmentid, objectid, segmentnumber, segmentstart, segmentend, segmentstartabs, segmentendabs) = cur.fetchone()
        return Multimediasegment(segmentid=segmentid, objectid=objectid, segmentnumber=segmentnumber,
                                 segmentstart=segmentstart, segmentend=segmentend, segmentstartabs=segmentstartabs,
                                 segmentendabs=segmentendabs)


def segmentinfos_post(body):  # noqa: E501
    """Get the information of multiple segments.

     # noqa: E501

    :param segmentinfos_post_request:
    :type segmentinfos_post_request: dict | bytes

    :rtype: Union[List[Multimediasegment], Tuple[List[Multimediasegment], int], Tuple[List[Multimediasegment], int, Dict[str, str]]
    """
    with get_connection(body['database']) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
                SELECT segmentid, objectid, segmentnumber, segmentstart, segmentend, segmentstartabs, segmentendabs  
                FROM cineast_segment WHERE segmentid = ANY(%s)""",
            (body['segmentids'],))
        results = cur.fetchall()

    segment_infos = [Multimediasegment(segmentid=segmentid, objectid=objectid, segmentnumber=segmentnumber,
                                       segmentstart=segmentstart, segmentend=segmentend,
                                       segmentstartabs=segmentstartabs,
                                       segmentendabs=segmentendabs) for
                     (segmentid, objectid, segmentnumber, segmentstart, segmentend, segmentstartabs, segmentendabs) in
                     results]

    return segment_infos


def querybyexample_post(body):  # noqa: E501
    """Get the nearest neighbors of a segment.

     # noqa: E501

    :param querybyexample_post_request:
    :type querybyexample_post_request: dict | bytes

    :rtype: Union[List[Scoredsegment], Tuple[List[Scoredsegment], int], Tuple[List[Scoredsegment], int, Dict[str, str]]
    """
    with get_connection(body['database']) as conn:
        cur = conn.cursor()
        cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(conn)

        limit = f'LIMIT {body["limit"]}' if 'limit' in body else ''

        if 'limit' in body:
            cur.execute('SET hnsw.ef_search = %s', (body['limit'],))

        cur.execute(
            f"""
            WITH query_feature AS (
                SELECT feature 
                FROM features_openclip 
                WHERE id = %s 
                LIMIT 1
            )
            SELECT 
                id, 
                (feature <=> (SELECT feature FROM query_feature)) AS distance
            FROM features_openclip
            WHERE id != %s
            ORDER BY distance
            {limit}
            """,
            (body['segmentid'], body['segmentid'])
        )

        results = cur.fetchall()
        scored_segments = [Scoredsegment(segmentid=segmentid, score=1 - distance) for (segmentid, distance) in results]
    return scored_segments


def segmentbytime_post(body):  # noqa: E501
    """Get the segment ID for a given timestamp and object.

     # noqa: E501

    :param segmentbytime_post_request:
    :type segmentbytime_post_request: dict | bytes

    :rtype: Union[SegmentbytimePost200Response, Tuple[SegmentbytimePost200Response, int], Tuple[SegmentbytimePost200Response, int, Dict[str, str]]
    """
    with get_connection(body['database']) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT segmentid 
            FROM cineast_segment 
            WHERE objectid = %s 
            AND %s BETWEEN segmentstartabs AND segmentendabs;
            """,
            (body['objectid'], body['timestamp'])
        )

        result = cur.fetchone()

    if result:
        return SegmentbytimePost200Response(segmentid=result[0])

    return {}, 404  # No matching segment found
