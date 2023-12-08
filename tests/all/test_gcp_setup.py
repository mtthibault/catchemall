import os

import pytest
from google.cloud import storage

from prediction.params import *
from tests.test_base import TestBase


class TestGcpSetup(TestBase):
    def test_setup_key_env(self):
        """
        verify that `$GOOGLE_APPLICATION_CREDENTIALS` is defined
        """

        # verify env var presence
        assert os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), "GCP environment variable not defined"

    def test_setup_key_path(self):
        """
        verify that `$GOOGLE_APPLICATION_CREDENTIALS` points to an existing file
        """

        service_account_key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # verify env var path existence
        with open(service_account_key_path, "r") as file:
            content = file.read()

        assert content is not None

    def test_code_get_wagon_project(self):
        """
        retrieve default gcp project id with code
        """
        # get default project id
        client = storage.Client(project=GCP_PROJECT_WAGON)
        project_id = client.project

        assert project_id is not None

    def test_code_get_project(self):
        """
        retrieve default gcp project id with code
        """
        # get default project id
        client = storage.Client()
        project_id = client.project

        assert project_id is not None

    def test_setup_project_id(self):
        """
        verify that the provided project id is correct
        """
        env_project_id = GCP_PROJECT
        # get default project id
        client = storage.Client()
        project_id = client.project

        assert env_project_id == project_id, f"GCP_PROJECT environmental variable differs from the activated GCP project ID"

    def test_setup_bucket_name(self):
        """
        verify that the provided bucket exists with correct name
        """

        env_bucket_name = BUCKET_NAME
        client = storage.Client()
        try:
            client.get_bucket(env_bucket_name, timeout=10)
        except:
            assert False, f"Your bucket named after your .env variable 'BUCKET_NAME' ({env_bucket_name}) could not be found. "
