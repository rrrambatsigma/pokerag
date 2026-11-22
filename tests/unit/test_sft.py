"""B5 dataset: extraction, rendering, variants and leak gates."""

from __future__ import annotations

from pokerag.sft import facts as F
from pokerag.sft.dataset import (
    ABSTAIN_ANSWER,
    Item,
    build_examples,
    build_items,
    split_items,
)
from pokerag.sft.leak import (
    check_text,
    entity_collisions,
    longest_common,
    normalise,
    question_collisions,
    word_ngrams,
)
from pokerag.sft.render import render, render_answer, render_evidence

DUAL_LEAD = (
    "Geodude (Japanese: イシツブテ Isitsubute) is a dual-type Rock / Ground Pokémon "
    "introduced in Generation I.\nIt evolves into Graveler starting at level 25, "
    "which evolves into Golem when traded."
)
SINGLE_LEAD = (
    "Plusle (Japanese: プラスル Prasle) is an Electric-type Pokémon introduced in "
    "Generation III."
)
INFOBOX = (
    "Plusle | #0311\nElectric\nPlus | Minus\nCatch rate 200 (26.1%)\n"
    "Egg Groups Fairy and Field | Hatch time 20 cycles\n"
)


def chunks(lead: str = SINGLE_LEAD, box: str = INFOBOX):
    return [([], lead), (["Infobox"], box)]


def test_extracts_single_and_dual_type():
    single = F.extract("Plusle", "d1", 1, chunks())
    assert F.find(single, F.TYPE).values == ("Electric",)

    dual = F.extract("Geodude", "d2", 2, chunks(lead=DUAL_LEAD))
    assert F.find(dual, F.TYPE).values == ("Rock", "Ground")


def test_extracts_catch_rate_and_egg_groups():
    found = {fact.kind: fact for fact in F.extract("Plusle", "d1", 1, chunks())}
    assert found[F.CATCH_RATE].values == ("200", "26.1")
    assert found[F.EGG_GROUP].values == ("Fairy", "Field")


def test_evolution_keeps_levels_only():
    found = {fact.kind: fact for fact in F.extract("Geodude", "d2", 2, chunks(lead=DUAL_LEAD))}
    assert found[F.EVOLUTION].values == ("Graveler", "25")


def test_verify_flags_a_stale_value():
    fact = F.find(F.extract("Plusle", "d1", 1, chunks()), F.CATCH_RATE)
    assert F.verify(fact, chunks()) is not None
    stale = F.Fact(fact.kind, fact.entity, fact.doc_id, fact.revid, fact.locator, ("10", "26.1"))
    assert F.verify(stale, chunks()) is None


def test_render_uses_the_right_article():
    fact = F.Fact(F.TYPE, "Plusle", "d1", 1, "Lead", ("Electric",))
    assert render_evidence(fact) == "Plusle is an Electric-type Pokémon."
    fact = F.Fact(F.TYPE, "Stantler", "d2", 1, "Lead", ("Normal",))
    assert render_evidence(fact) == "Stantler is a Normal-type Pokémon."


def test_render_dual_and_catch_rate():
    dual = F.Fact(F.TYPE, "Geodude", "d1", 1, "Lead", ("Rock", "Ground"))
    assert render_evidence(dual) == "Geodude has two types, Rock and Ground."
    assert render_answer(dual) == "Geodude has two types, Rock and Ground"

    catch = F.Fact(F.CATCH_RATE, "Plusle", "d1", 1, "Infobox", ("200", "26.1"))
    assert render_answer(catch) == "Plusle has a catch rate of 200 (26.1%)"


def test_render_question_rotates():
    fact = F.Fact(F.CATCH_RATE, "Plusle", "d1", 1, "Infobox", ("200", "26.1"))
    assert render(fact, 0).question == "What is the catch rate of Plusle?"
    assert render(fact, 1).question == "What catch rate does Plusle have?"


def _facts():
    return {
        "d1": {F.TYPE: F.Fact(F.TYPE, "Abra", "d1", 1, "Lead", ("Psychic",))},
        "d2": {F.TYPE: F.Fact(F.TYPE, "Kadabra", "d2", 1, "Lead", ("Psychic",))},
        "d3": {F.CATCH_RATE: F.Fact(F.CATCH_RATE, "Machop", "d3", 1, "Infobox", ("180", "23.5"))},
    }


def test_build_items_respects_quota_and_uses_each_doc_once():
    items = build_items(_facts(), {F.TYPE: 2, F.CATCH_RATE: 1}, seed=7)
    assert len(items) == 3
    assert len({item.doc_id for item in items}) == 3
    assert {item.item_id for item in items} == {"sft0001", "sft0002", "sft0003"}


def test_build_items_is_deterministic():
    first = [item.as_dict() for item in build_items(_facts(), {F.TYPE: 2, F.CATCH_RATE: 1}, seed=7)]
    second = [item.as_dict() for item in build_items(_facts(), {F.TYPE: 2, F.CATCH_RATE: 1}, seed=7)]
    assert first == second


def _items(count: int = 6) -> list[Item]:
    items = []
    for index in range(count):
        items.append(
            Item(
                item_id=f"i{index}",
                entity=f"Mon{index}",
                kind=F.CATCH_RATE,
                question=f"What is the catch rate of Mon{index}?",
                evidence=f"Mon{index} has a catch rate of {index + 1}, which equals 1.0%.",
                answer=f"Mon{index} has a catch rate of {index + 1} (1.0%)",
                doc_id=f"d{index}",
                revid=1,
                locator="Infobox",
                values=(str(index + 1), "1.0"),
            )
        )
    items.append(
        Item(
            item_id="other",
            entity="Odd",
            kind=F.TYPE,
            question="What type is Odd?",
            evidence="Odd is a Normal-type Pokémon.",
            answer="Odd is a Normal-type Pokémon",
            doc_id="dx",
            revid=1,
            locator="Lead",
            values=("Normal",),
        )
    )
    items.append(
        Item(
            item_id="other2",
            entity="Oddish",
            kind=F.TYPE,
            question="What type is Oddish?",
            evidence="Oddish has two types, Grass and Poison.",
            answer="Oddish has two types, Grass and Poison",
            doc_id="dy",
            revid=1,
            locator="Lead",
            values=("Grass", "Poison"),
        )
    )
    items.append(
        Item(
            item_id="other3",
            entity="Gloom",
            kind=F.TYPE,
            question="What type is Gloom?",
            evidence="Gloom has two types, Grass and Poison.",
            answer="Gloom has two types, Grass and Poison",
            doc_id="dz",
            revid=1,
            locator="Lead",
            values=("Grass", "Poison"),
        )
    )
    return items


def test_examples_have_three_variants_with_a_pointer_to_the_answer():
    items = _items()
    by_id = {item.item_id: item for item in items}
    examples = build_examples(items, seed=7)
    assert len(examples) == len(items) * 3
    for example in examples:
        if example.variant == "abstain":
            assert example.abstain is True
            assert example.citations == ()
            assert example.answer == ABSTAIN_ANSWER
            continue
        assert example.abstain is False
        index = example.citations[0] - 1
        own = by_id[example.item_id]
        assert example.evidence[index][2] == own.evidence
        assert example.answer == f"{own.answer} [{example.citations[0]}]"


def test_split_keeps_items_whole():
    items = _items(10)
    train, val = split_items(items, val_every=3)
    assert not {item.item_id for item in train} & {item.item_id for item in val}
    assert len(train) + len(val) == len(items)


def test_ngram_gate_flags_a_copied_sentence():
    corpus = [(["Evidence"], "It evolves into Ivysaur starting at level 16, which is early.")]
    copied = "It evolves into Ivysaur starting at level 16, which is early."
    assert check_text(copied, corpus).copied is True

    ours = "Ivysaur arrives at level 16."
    assert check_text(ours, corpus).copied is False


def test_longest_common_and_ngrams():
    assert longest_common("abcdef", "xxabcdefyy") == 6
    assert word_ngrams("one two three", n=3) == {("one", "two", "three")}
    assert word_ngrams("one two", n=3) == set()


def test_entity_and_question_collisions():
    items = _items(2)
    assert entity_collisions(items, ["Mon0"]) == [("i0", "Mon0")]
    assert question_collisions(items, [items[0].question]) == [("i0", items[0].question)]
    assert normalise("Nidoran♀") == "nidoran"
