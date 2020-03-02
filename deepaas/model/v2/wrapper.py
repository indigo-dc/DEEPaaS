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
import collections
import concurrent.futures
import contextlib
import datetime
import functools
import multiprocessing
import multiprocessing.pool
import os
import signal
import tempfile

from aiohttp import web
import marshmallow
import marshmallow.utils
import webargs.core
from webargs import fields

from oslo_config import cfg
from oslo_log import log

LOG = log.getLogger(__name__)

CONF = cfg.CONF


UploadedFile = collections.namedtuple("UploadedFile", ("name",
                                                       "filename",
                                                       "content_type",
                                                       "original_filename"))
"""Class to hold uploaded field metadata when passed to model's methods

.. py:attribute:: name

   Name of the argument where this file is being sent.

.. py:attribute:: filename

   Complete file path to the temporary file in the filesyste,

.. py:attribute:: content_type

   Content-type of the uploaded file

.. py:attribute:: original_filename

   Filename of the original file being uploaded.
"""

# set defaults to None, mainly for compatibility (vkoz)
UploadedFile.__new__.__defaults__ = (None, None, None, None)


class ModelWrapper(object):
    """Class that will wrap the loaded models before exposing them.

    Whenever a model is loaded it will be wrapped with this class to create a
    wrapper object that will handle the calls to the model's methods so as to
    handle non-existent method exceptions.

    :param name: Model name
    :param model: Model object
    :raises HTTPInternalServerError: in case that a model has defined
        a reponse schema that is nod JSON schema valid (DRAFT 4)
    """
    def __init__(self, name, model_obj, app):
        self.name = name
        self.model_obj = model_obj
        self._app = app

        self._loop = asyncio.get_event_loop()

        self._predict_workers = CONF.predict_workers
        self._predict_executor = self._init_predict_executor()

        self._train_workers = CONF.train_workers
        self._train_executor = self._init_train_executor()

        self._setup_cleanup()

        schema = getattr(self.model_obj, "schema", None)

        if isinstance(schema, dict):
            try:
                schema = marshmallow.Schema.from_dict(
                    schema,
                    name="ModelPredictionResponse"
                )
                self.has_schema = True
            except Exception as e:
                LOG.exception(e)
                raise web.HTTPInternalServerError(
                    reason=("Model defined schema is invalid, "
                            "check server logs.")
                )
        elif schema is not None:
            try:
                if issubclass(schema, marshmallow.Schema):
                    self.has_schema = True
            except TypeError:
                raise web.HTTPInternalServerError(
                    reason=("Model defined schema is invalid, "
                            "check server logs.")
                )
        else:
            self.has_schema = False

        self.response_schema = schema

    def _setup_cleanup(self):
        self._app.on_cleanup.append(self._close_executors)

    async def _close_executors(self, app):
        self._train_executor.shutdown()
        self._predict_executor.shutdown()

    def _init_predict_executor(self):
        n = self._predict_workers
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=n)
        return executor

    def _init_train_executor(self):
        n = self._train_workers
        executor = CancellablePool(max_workers=n)
        return executor

    @contextlib.contextmanager
    def _catch_error(self):
        name = self.name
        try:
            yield
        except AttributeError:
            raise web.HTTPNotImplemented(
                reason=("Not implemented by underlying model (loaded '%s')" %
                        name)
            )
        except NotImplementedError:
            raise web.HTTPNotImplemented(
                reason=("Model '%s' does not implement this functionality" %
                        name)
            )
        except Exception as e:
            LOG.error("An exception has happened when calling method on "
                      "'%s' model." % name)
            LOG.exception(e)
            if isinstance(e, web.HTTPException):
                raise e
            else:
                raise web.HTTPInternalServerError(reason=e)

    def validate_response(self, response):
        """Validate a response against the model's response schema, if set.

        If the wrapped model has defined a ``response`` attribute we will
        validate the response that

        :param response: The reponse that will be validated.
        :raises exceptions.InternalServerError: in case the reponse cannot be
            validated.
        """
        if self.has_schema is not True:
            raise web.HTTPInternalServerError(
                reason=("Trying to validate against a schema, but I do not "
                        "have one defined")
            )

        try:
            self.response_schema().load(response)
        except marshmallow.ValidationError as e:
            LOG.exception(e)
            raise web.HTTPInternalServerError(
                reason="ERROR validating model response, check server logs."
            )
        except Exception as e:
            LOG.exception(e)
            raise web.HTTPInternalServerError(
                reason="Unknown ERROR validating response, check server logs."
            )

        return True

    def get_metadata(self):
        """Obtain model's metadata.

        If the model's metadata cannot be obtained because it is not
        implemented, we will provide some generic information so that the
        call does not fail.

        :returns dict: dictionary containing model's metadata
        """
        try:
            d = self.model_obj.get_metadata()
        except (NotImplementedError, AttributeError):
            d = {
                "id": "0",
                "name": self.name,
                "description": ("Could not load description from "
                                "underlying model (loaded '%s')" % self.name),
            }
        return d

    def _run_in_predict_pool(self, func, *args, **kwargs):
        async def task(fn):
            return await self._loop.run_in_executor(self._predict_executor, fn)

        return self._loop.create_task(
            task(functools.partial(func, *args, **kwargs))
        )

    def _run_in_train_pool(self, func, *args, **kwargs):
        fn = functools.partial(func, *args, **kwargs)
        ret = self._loop.create_task(
            self._train_executor.apply(fn)
        )
        return ret

    async def warm(self):
        """Warm (i.e. load, initialize) the underlying model.

        This method is called automatically when the model is loaded. You
        should use this method to initialize the model so that it is ready for
        the first prediction.

        The model receives no arguments.
        """
        try:
            func = self.model_obj.warm
        except AttributeError:
            LOG.debug("Cannot warm (initialize) model '%s'" % self.name)
            return

        run = self._loop.run_in_executor
        executor = self._predict_executor
        n = self._predict_workers
        try:
            LOG.debug("Warming '%s' model with %s workers" % (self.name, n))
            fs = [run(executor, func) for i in range(0, n)]
            await asyncio.gather(*fs)
            LOG.debug("Model '%s' has been warmed" % self.name)
        except NotImplementedError:
            LOG.debug("Cannot warm (initialize) model '%s'" % self.name)

    def predict(self, *args, **kwargs):
        """Perform a prediction on wrapped model's ``predict`` method.

        :raises HTTPNotImplemented: If the method is not
            implemented in the wrapper model.
        :raises HTTPInternalServerError: If the call produces
            an error
        :raises HTTPException: If the call produces an
            error, already wrapped as a HTTPException
        """
        for key, val in kwargs.items():
            if isinstance(val, web.FileField):
                fd, name = tempfile.mkstemp()
                fd = os.fdopen(fd, "w+b")
                fd.write(val.file.read())
                fd.close()
                aux = UploadedFile(
                    name=val.name,
                    filename=name,
                    content_type=val.content_type,
                    original_filename=val.filename
                )
                kwargs[key] = aux
                # FIXME(aloga); cleanup of tmpfile here

        with self._catch_error():
            return self._run_in_predict_pool(
                self.model_obj.predict, *args, **kwargs
            )

    def train(self, *args, **kwargs):
        """Perform a training on wrapped model's ``train`` method.

        :raises HTTPNotImplemented: If the method is not
            implemented in the wrapper model.
        :raises HTTPInternalServerError: If the call produces
            an error
        :raises HTTPException: If the call produces an
            error, already wrapped as a HTTPException
        """

        with self._catch_error():
            return self._run_in_train_pool(
                self.model_obj.train, *args, **kwargs
            )

    def get_train_args(self):
        """Add training arguments into the training parser.

        :param parser: an argparse like object

        This method will call the wrapped model ``add_train_args``. If the
        underlying methid implements the DEPRECATED way of passing arguments
        we will try to load them from there.
        """
        try:
            args = self.model_obj.get_train_args()
            args = self._convert_old_args(args)
            return args
        except (NotImplementedError, AttributeError):
            return {}

    def get_predict_args(self):
        """Add predict arguments into the predict parser.

        :param parser: an argparse like object

        This method will call the wrapped model ``get_predict_args``. If the
        method does not exist, but the wrapped model implements the DEPRECATED
        ``get_test_args`` we will try to load the arguments from there.
        """
        try:
            args = self.model_obj.get_predict_args()
        except (NotImplementedError, AttributeError):
            try:
                args = self.model_obj.get_test_args()
                args = self._convert_old_args(args)
            except (NotImplementedError, AttributeError):
                args = {}
        return self._args_to_schemas(args)

    @staticmethod
    def _args_to_schemas(args):
        """Transform from argument dictionaries to webargs schemas.

        This method receives a dictionary whose elements are webargs.Fields and
        returns a list of schemas, grouped by location, so that these fields
        can be populated into the API using the correct use_args and use_kwargs
        methods.

        We cannot simply return a single dictionary due to the breaking change
        introduced in webargs version 6.0.0 that removed the support for
        defining locations inse the individual fields:
        https://webargs.readthedocs.io/en/latest/changelog.html#b1-2020-01-06
        """
        # Available locati
        locations = {
            "query": {},
            "json": {},
            "form": {},
            "headers": {},
            "cookies": {},
            "files": {},
        }

        for name, arg in args.items():
            if getattr(arg, "location", None):
                if arg.location in ["query", "querystring"]:
                    locations["query"][name] = arg
                else:
                    locations[arg.location][name] = arg
            elif name == "accept":  # PEBKAC but we need to handle it
                locations["headers"][name] = arg
            else:
                locations["query"][name] = arg

        accept = locations["headers"].get("accept", None)
        if accept:
            accept.validate.choices.append("*/*")
            # If no default value use first possible choice:
            if isinstance(accept.missing, marshmallow.utils._Missing):
                accept.missing = accept.validate.choices[0]

        import pprint
        pprint.pprint(locations)
        schemas = {}
        for l, d in locations.items():
            s = webargs.core.dict2schema(d)
            s.opts.ordered = True
            schemas[l] = s
        return schemas

    def _convert_old_args(self, args):
        aux = {}
        for k, v in args.items():
            if isinstance(v, dict):
                LOG.warning("Loading arguments using the old and DEPRECATED "
                            "return value (i.e. an plain Python dictionary. "
                            "You should move to the new return value (i.e. a "
                            "webargs.fields dictionary as soon as possible. "
                            "All the loaded arguments will be converted to "
                            "strings. This is only supported for backwards "
                            "compatibility and may lead to unexpected errors. "
                            "Argument raising this warningr: '%s'", k)

                v = fields.Str(
                    missing=v.get("default"),
                    description=v.get("help"),
                    required=v.get("required"),
                )
            aux[k] = v
        return aux


class NonDaemonProcess(multiprocessing.context.SpawnProcess):
    """Processes must use 'spawn' instead of 'fork' (which is the default
    in Linux) in order to work CUDA [1] or Tensorflow [2].

    [1] https://pytorch.org/docs/stable/notes/multiprocessing.html
    #cuda-in-multiprocessing
    [2] https://github.com/tensorflow/tensorflow/issues/5448
    #issuecomment-258934405
    """
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NonDaemonPool(multiprocessing.pool.Pool):
    # Based on https://stackoverflow.com/questions/6974695/
    Process = NonDaemonProcess


class CancellablePool(object):
    def __init__(self, max_workers=None):
        self._free = {self._new_pool() for _ in range(max_workers)}
        self._working = set()
        self._change = asyncio.Event()

    def _new_pool(self):
        return NonDaemonPool(1, context=multiprocessing.get_context('spawn'))

    async def apply(self, fn, *args):
        """
        Like multiprocessing.Pool.apply_async, but:
         * is an asyncio coroutine
         * terminates the process if cancelled
        """
        while not self._free:
            await self._change.wait()
            self._change.clear()
        pool = usable_pool = self._free.pop()
        self._working.add(pool)

        loop = asyncio.get_event_loop()
        fut = loop.create_future()

        def _on_done(obj):
            ret = {'output': obj,
                   'finish_date': str(datetime.datetime.now())}
            loop.call_soon_threadsafe(fut.set_result, ret)

        def _on_err(err):
            loop.call_soon_threadsafe(fut.set_exception, err)

        pool.apply_async(fn, args, callback=_on_done, error_callback=_on_err)

        try:
            return await fut
        except asyncio.CancelledError:
            # This is ugly, but since our pools only have one slot we can
            # kill the process before termination
            try:
                pool._pool[0].kill()
            except AttributeError:
                os.kill(pool._pool[0].pid,
                        signal.SIGKILL)
            pool.terminate()
            usable_pool = self._new_pool()
        finally:
            self._working.remove(pool)
            self._free.add(usable_pool)
            self._change.set()

    def shutdown(self):
        for p in self._working:
            p.terminate()
        self._free.clear()
