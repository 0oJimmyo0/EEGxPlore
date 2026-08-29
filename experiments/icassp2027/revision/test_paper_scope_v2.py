"""Contract test for the final 12-cell ICASSP confirmatory manifest."""

from __future__ import annotations

from audit_paper_scope_v2 import DEFAULT_TABLE, audit


def main() -> None:
    result = audit(DEFAULT_TABLE)
    assert result['passed'], result['errors']
    assert result['confirmatory_rows'] == 12
    assert result['development_seed_excluded'] == '42'
    print('paper scope v2 contract: PASS')


if __name__ == '__main__':
    main()
