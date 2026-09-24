"""서로 독립인 기상청 조회를 동시에 돌리는 도우미.

예전에는 선택 항목(미세먼지·꽃가루·영향예보 등)을 하나씩 차례로 받았다. 기상청이
504 로 느릴 때는 항목마다 제한 시간(30초)을 기다려 첫 조회가 10분을 넘겼고, HA 가
시작하면서 그 설정을 취소해(setup_error) 다시 시도하지 않는 일이 생겼다.

- 한 번에 OPTIONAL_CONCURRENCY 개까지 동시에 보낸다.
- 첫 조회(first=True)에서는 FIRST_REFRESH_DEADLINE 초 안에 끝나지 않은 항목을 취소하고
  skipped(key) 값으로 채운다. 그 항목은 다음 정기 갱신 때 다시 받는다.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

_LOGGER = logging.getLogger(__name__)

OPTIONAL_CONCURRENCY = 4        # 기상청에 한꺼번에 보내는 요청 수
FIRST_REFRESH_DEADLINE = 45     # 첫 조회에서 선택 항목을 기다리는 최대 초


async def run_concurrently(
    jobs: dict[str, Callable[[], Awaitable[Any]]],
    *,
    first: bool,
    skipped: Callable[[str], Any],
    concurrency: int = OPTIONAL_CONCURRENCY,
    deadline: float = FIRST_REFRESH_DEADLINE,
) -> dict[str, Any]:
    """jobs 의 각 조회를 동시에 실행하고 {key: 결과} 를 jobs 순서대로 돌려준다.

    예상하지 못한 예외는 예전처럼 그대로 올린다(조회 함수가 알려진 오류는 스스로 처리한다).
    """
    sem = asyncio.Semaphore(concurrency)

    async def one(fn: Callable[[], Awaitable[Any]]) -> Any:
        async with sem:
            return await fn()

    tasks = {key: asyncio.ensure_future(one(fn)) for key, fn in jobs.items()}
    if not tasks:
        return {}
    try:
        _done, pending = await asyncio.wait(tasks.values(), timeout=deadline if first else None)
    except asyncio.CancelledError:                  # 종료 등으로 갱신 자체가 취소되면 함께 정리
        for t in tasks.values():
            t.cancel()
        raise
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
        late = [k for k, t in tasks.items() if t in pending]
        _LOGGER.info("첫 조회에서 %d초 안에 받지 못한 %d개 항목은 다음 갱신 때 받습니다: %s",
                     deadline, len(late), ", ".join(late))
    return {k: (skipped(k) if t in pending else t.result()) for k, t in tasks.items()}
