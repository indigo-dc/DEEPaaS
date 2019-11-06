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

import asyncio
import datetime
import uuid

from aiohttp import web
import aiohttp_apispec
from oslo_log import log
from webargs import aiohttpparser
import webargs.core

from deepaas.api.v2 import responses
from deepaas import model

LOG = log.getLogger("deepaas.api.v2.train")


def _get_handler(model_name, model_obj):  # noqa
    args = webargs.core.dict2schema(model_obj.get_train_args())

    class Handler(object):
        model_name = None
        model_obj = None

        def __init__(self, model_name, model_obj):
            self.model_name = model_name
            self.model_obj = model_obj
            self._trainings = {}

        def build_train_response(self, uuid_):
            training = self._trainings.get(uuid_, None)

            if not training:
                return

            ret = {}
            ret["date"] = training["date"]
            ret["uuid"] = uuid_

            if training["task"].cancelled():
                ret["status"] = "cancelled"
            elif training["task"].done():
                exc = training["task"].exception()
                if exc:
                    ret["status"] = "error"
                    ret["message"] = "%s" % exc
                else:
                    ret["status"] = "done"
            else:
                ret["status"] = "running"
            return ret

        @aiohttp_apispec.docs(
            tags=["models"],
            summary="Retrain model with available data"
        )
        @aiohttp_apispec.querystring_schema(args)
        @aiohttpparser.parser.use_args(args)
        async def post(self, request, args):
            uuid_ = uuid.uuid4().hex
            train_task = self.model_obj.train(**args)
            self._trainings[uuid_] = {
                "date": str(datetime.datetime.now()),
                "task": train_task,
            }
            ret = self.build_train_response(uuid_)
            return web.json_response(ret)

        @aiohttp_apispec.docs(
            tags=["models"],
            summary="Cancel a running training"
        )
        async def delete(self, request):
            uuid_ = request.match_info["uuid"]
            training = self._trainings.get(uuid_, None)
            if not training:
                raise web.HTTPNotFound()
            training["task"].cancel()
            try:
                await asyncio.wait_for(training["task"], 5)
            except asyncio.TimeoutError:
                pass
            LOG.info("Training %s has been cancelled" % uuid_)
            ret = self.build_train_response(uuid_)
            return web.json_response(ret)

        @aiohttp_apispec.docs(
            tags=["models"],
            summary="Get a list of trainings (running or completed)"
        )
        @aiohttp_apispec.response_schema(responses.TrainingList(), 200)
        async def index(self, request):
            ret = []
            for uuid_, training in self._trainings.items():
                aux = self.build_train_response(uuid_)
                ret.append(aux)

            return web.json_response(ret)

        @aiohttp_apispec.docs(
            tags=["models"],
            summary="Get status of a training"
        )
        @aiohttp_apispec.response_schema(responses.Training(), 200)
        async def get(self, request):
            uuid_ = request.match_info["uuid"]
            ret = self.build_train_response(uuid_)
            if ret:
                return web.json_response(ret)
            raise web.HTTPNotFound()

    return Handler(model_name, model_obj)


def setup_routes(app):
    # In the next lines we iterate over the loaded models and create the
    # different resources for each model. This way we can also load the
    # expected parameters if needed (as in the training method).
    for model_name, model_obj in model.V2_MODELS.items():
        hdlr = _get_handler(model_name, model_obj)
        app.router.add_post("/models/%s/train" % model_name, hdlr.post)
        app.router.add_get("/models/%s/train" % model_name, hdlr.index)
        app.router.add_get("/models/%s/train/{uuid}" % model_name, hdlr.get)
        app.router.add_delete(
            "/models/%s/train/{uuid}" % model_name,
            hdlr.delete
        )