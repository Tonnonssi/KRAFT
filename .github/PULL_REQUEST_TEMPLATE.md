<!--
개인 프로젝트 사용 가이드
- 5분 이내 작성이 목표입니다.
- 비어 있으면 "해당 없음"으로 표기하세요.
- 체크박스는 [x] 로 표시합니다.
-->

# 제목
<!-- 가능하면 Conventional Commits: feat(scope): 요약 -->
feat(...): ...

## PR Type (필수)
- [ ] Feature
- [ ] Bug Fix
- [ ] Refactor
- [ ] Performance
- [ ] Docs
- [ ] Infra/CI
- [ ] Data
- [ ] Experiment
- [ ] Hotfix

## Summary (필수)
이 PR이 해결하는 목표를 1~2문장으로 요약합니다.

## Changes (핵심만)
- 무엇을 바꿨는지 항목으로 요약
- 선택한 접근/대안 간단 비교(있으면)

## Test Plan (필수)
- 자동/수동 검증 방법 2~5줄
- 재현 케이스(버그일 경우)와 기대 동작

## Evidence
스크린샷/로그/벤치마크 등 확인 가능한 증거(있으면 첨부)

## Risk & Backout
- 파괴적 변경: [ ] 없음 / [ ] 있음(아래 설명)
- 이상 시 되돌리기 방법(커밋/태그, 간단 설명)

## Follow-ups / TODO
합치고 나중에 할 일(빚/개선 포인트) 0~3개

## Meta (선택)
- 시간 투자(대략): ~ h
- 관련 이슈/메모 링크: …

---

<!-- 유형별 짧은 보조 섹션: 해당되는 경우만 기입 -->

<details>
<summary>Experiment 선택 시</summary>

### 가설 / 목표
- …

### 설정(모델/데이터/HP)
- …

### 결과 요약
- …

</details>

<details>
<summary>Infra/CI 선택 시</summary>

### 변경 요점
- Docker/Action/빌드 스크립트 …

### 검증
- 로컬/CI 로그 요약

</details>