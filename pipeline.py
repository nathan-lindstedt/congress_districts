#%%
# Import libraries
import joblib
import os
import urllib

import numpy as np
import pandas as pd
import sklearn

from steps.preprocess import preprocess
from steps.train import train
from steps.test import test
# from steps.register import register
# from steps.deploy import deploy

from sagemaker.s3 import S3Uploader
from sagemaker.session import Session
from sagemaker import get_execution_role
from sagemaker.workflow.function_step import step
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterFloat, ParameterInteger
)

import xgboost

#%%
# Import S3 data
def import_data_s3_uri(bucket_name: str) -> str:
    s3_prefix = "data"
    import_data_s3_uri = f"s3://{bucket_name}/{s3_prefix}"

    return import_data_s3_uri

#%%
# Create pipeline steps
def create_steps(role, input_data_s3_uri, bucket_name,
                 model_package_group_name, model_approval_status,
                 deploy_model_parameter):

    preprocess_result = step(preprocess, name="Preprocess", keep_alive_period_in_seconds=300)(
        input_data_s3_uri)

    train_result = step(train, name="Train", keep_alive_period_in_seconds=300)(
        X_train=preprocess_result[0], y_train=preprocess_result[1], 
        X_val=preprocess_result[2], y_val=preprocess_result[3])

    test_result = step(test, name="Evaluate", keep_alive_period_in_seconds=300)(
        featurizer_model=preprocess_result[6], booster=train_result, 
        X_test=preprocess_result[4], y_test=preprocess_result[5]
    )

    # register_result = step(register, name="Register", keep_alive_period_in_seconds=300)(role,
    #     featurizer_model=preprocess_result[6], booster=train_result, 
    #     bucket_name=bucket_name, model_report_dict=test_result,
    #     model_package_group_name=model_package_group_name,
    #     model_approval_status=model_approval_status)

    # deploy_result = step(deploy, name="Deploy", keep_alive_period_in_seconds=300)(role,
    #     model_package_arn=register_result, deploy_model=deploy_model_parameter)

    return [test_result]

#%%
if __name__ == "__main__":
    os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = os.getcwd()

    role = get_execution_role()
    bucket_name = Session().default_bucket()
    pipeline_name = "concrete-abstraction-pipeline"
    model_package_group_name = "concrete-abstraction-model-package-group"
    model_approval_status = "PendingManualApproval"

    deploy_model_parameter = ParameterBoolean(
        name="deploy_model", default_value=True
    )
    
    input_data_s3_uri = import_data_s3_uri(bucket_name)
    
    steps=create_steps(role, input_data_s3_uri, bucket_name, 
                       model_package_group_name, model_approval_status,
                       deploy_model_parameter)

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[deploy_model_parameter],
        steps=steps
    )

    pipeline.upsert(role_arn=role)
    pipeline.start()

#%%
