import unittest

from flask import json

from ferelight.models.multimediaobject import Multimediaobject  # noqa: E501
from ferelight.test import BaseTestCase


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    def test_objectinfo_objectid_get(self):
        """Test case for objectinfo_objectid_get

        Get the information of an object.
        """
        headers = { 
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/objectinfo/{objectid}'.format(objectid='objectid_example'),
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
