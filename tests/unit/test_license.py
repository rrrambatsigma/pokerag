"""Licence handling is a governance requirement, not a comment in a README."""

from __future__ import annotations


def test_nc_sa_content_is_never_allowed_for_fine_tuning(cfg):
    license_info = cfg.license_for(123456)

    assert license_info.name == "CC-BY-NC-SA-2.5"
    assert license_info.allow_retrieval is True
    assert license_info.allow_redistribute is False
    assert license_info.allow_ft_training is False
    assert "123456" in license_info.attribution


def test_attribution_keeps_the_revision(cfg):
    assert cfg.license_for(99).attribution == "Bulbapedia, CC BY-NC-SA 2.5, revision 99"
