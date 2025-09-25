# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from comps import OpeaComponentLoader


class OpeaText2QueryLoader(OpeaComponentLoader):

    async def db_connection_check(self, *args, **kwargs):
        return await self.component.db_connection_check(*args, **kwargs)
