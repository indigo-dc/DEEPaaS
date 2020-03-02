# -*- coding: utf-8 -*-

# Copyright 2018 Spanish National Research Council (CSIC)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from aiohttp import web
import aiohttp_apispec
import marshmallow
from webargs import aiohttpparser
import webargs.core

from deepaas.api.v2 import responses
from deepaas.api.v2 import utils
from deepaas import model


def _get_model_response(model_name, model_obj):
    response_schema = model_obj.response_schema

    if response_schema is not None:
        return response_schema

    return responses.Prediction


def _get_handler(model_name, model_obj):
    hdlr_args = model_obj.get_predict_args()
#    accept = aux.get("accept", None)
#    if accept:
#        accept.validate.choices.append("*/*")
#        # If no default value use first possible choice:
#        if isinstance(accept.missing, marshmallow.utils._Missing):
#            accept.missing = accept.validate.choices[0]
#        accept.location = "headers"
#
#    hdlr_args = webargs.core.dict2schema(aux)
#    hdlr_args.opts.ordered = True

#    print(dir(hdlr_args["headers"]))
#    accept = hdlr_args["headers"].accept
    response = _get_model_response(model_name, model_obj)

    class Handler(object):
        model_name = None
        model_obj = None

        def __init__(self, model_name, model_obj):
            self.model_name = model_name
            self.model_obj = model_obj

        @aiohttp_apispec.docs(
            tags=["models"],
            summary="Make a prediction given the input data",
#            produces=accept.validate.choices if accept else None,
        )
        @aiohttp_apispec.querystring_schema(hdlr_args["query"])
        @aiohttp_apispec.json_schema(hdlr_args["json"])
        @aiohttp_apispec.form_schema(hdlr_args["form"])
        @aiohttp_apispec.headers_schema(hdlr_args["headers"])
        @aiohttp_apispec.cookies_schema(hdlr_args["cookies"])
        @aiohttp_apispec.request_schema(hdlr_args["files"], location="files")
        @aiohttp_apispec.response_schema(response(), 200)
        @aiohttp_apispec.response_schema(responses.Failure(), 400)
        async def post(self, request, wsk_args=None):
            print(request["data"])
            args = await aiohttpparser.parser.parse(hdlr_args, request)
            if wsk_args:
                args.update(wsk_args)
            task = self.model_obj.predict(**args)
            await task

            ret = task.result()

            accept = args.get("accept", "application/json")
            if accept != "application/json":
                response = web.Response(
                    body=ret,
                    content_type=accept,
                )
                return response
            if self.model_obj.has_schema:
                self.model_obj.validate_response(ret)
                return web.json_response(ret)

            return web.json_response({"status": "OK", "predictions": ret})

    return Handler(model_name, model_obj)


def setup_routes(app, enable=True):
    # In the next lines we iterate over the loaded models and create the
    # different resources for each model. This way we can also load the
    # expected parameters if needed (as in the training method).
    for model_name, model_obj in model.V2_MODELS.items():
        if enable:
            hdlr = _get_handler(model_name, model_obj)
        else:
            hdlr = utils.NotEnabledHandler()
        app.router.add_post(
            "/models/%s/predict/" % model_name,
            hdlr.post
        )
