"""Shared pytest fixtures for tests/unit/core/.

Covers Tool.from_callable() (src/atk/core/tool.py) and any related
unit-test files that may be split off under this directory.

pytest discovers this file automatically; no imports are needed in
individual test files.  The raw ``_example_*`` callables are also
importable directly for use with @pytest.mark.parametrize.

Coverage map
============

Original requirements
---------------------
  Req 1   str / int / float / bool, mixed required + optional
              → example_primitives
  Req 2a  Literal with string values only
              → example_string_literal
  Req 2b  Literal with non-string values → NotImplementedError
              → example_int_literal
              → example_mixed_literal
  Req 3   Concrete Enum subclass parameter
              → example_enum_param
  Req 4   Bare list + typed list[str] / list[int] / list[float] / list[bool]
              → example_list_params
  Req 5   Bare dict + typed dict[str, int] / dict[str, bool] / dict[str, float]
              → example_dict_params
  Req 6   Zero-argument callable
              → example_no_args

Gap fixtures
------------
  Gap 1   Optional[T] with None default → optional T in schema, not anyOf
              → example_optional_args
  Gap 3   Nested containers (list-of-list, dict-of-list)
              → example_nested_containers
  Gap 5   Degraded / missing docstrings
              → example_no_docstring          (no __doc__ at all)
              → example_description_only      (summary only, no Args section)
              → example_partial_args_doc      (Args section omits one param)
              → degraded_docstring_examples   (collected; use with parametrize)
  Gap 8   TypedDict parameter — flat and one-level-nested
              → example_flat_typed_dict
              → example_nested_typed_dict

Aggregate
---------
  all_examples   every callable above as an ordered list (smoke tests)

Design decisions
================
* Return annotations are NOT processed by the converter; all fixtures
  use -> None (or a simple primitive where needed for clarity).
* Optional[T] is supported only when the parameter's default is None.
  The converter emits a plain T schema and omits the parameter from
  ``required``.  anyOf / Union[A, B] (non-None) is out of scope.
* TypedDict is the ceiling for structured / nested parameter types.
  One level of nesting (TypedDict field whose value is itself a
  TypedDict) is supported; deeper recursion is not.
* Nested containers (list[list[T]], dict[str, list[T]]) are tested
  separately from TypedDict because the schema-emission logic differs.
* *args / **kwargs are silently skipped by the converter; no fixture
  is needed for that behaviour.
* The three degraded-docstring callables are individually wrapped in
  @pytest.fixture for targeted assertions AND collected into
  ``degraded_docstring_examples`` for a single parametrized smoke test.
"""

from __future__ import annotations

import enum
from typing import Literal, TypedDict

import pytest

# ------------------------------------------------------------------ #
# Supporting types                                                     #
# ------------------------------------------------------------------ #


class Severity(enum.Enum):
    """Urgency level used in alert-related tool calls."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BoundingBox(TypedDict):
    """Pixel-space rectangle; used as a flat TypedDict parameter (Gap 8)."""

    x: int
    y: int
    width: int
    height: int


class DetectionResult(TypedDict):
    """Object-detection hit; contains a nested BoundingBox.

    Gap 8, one-level nesting.
    """

    label: str
    confidence: float
    box: BoundingBox  # nested TypedDict — converter must recurse exactly one level


# ------------------------------------------------------------------ #
# Req 1 — Primitives (str / int / float / bool) + mixed required/optional
# ------------------------------------------------------------------ #


def _example_primitives(  # noqa: PLR0913 — test fixture requires all 8 params to exercise mixed required/optional
    message: str,
    count: int,
    threshold: float,
    verbose: bool,
    prefix: str = "info",
    retries: int = 3,
    jitter: float = 0.1,
    dry_run: bool = False,
) -> None:
    """Send a formatted log message to the configured output sink.

    Formats ``message`` with the supplied ``prefix`` and repeats the
    emission ``count`` times, stopping early if the internal error rate
    exceeds ``threshold``.  When ``verbose`` is ``True`` every attempt
    is echoed to stdout regardless of the sink configuration.

    Args:
        message: The human-readable body of the log entry.  Must not be
            empty; leading and trailing whitespace is stripped before
            transmission.
        count: Number of times to emit the message.  Must be a positive
            integer.
        threshold: Maximum tolerated error rate in the range ``[0.0,
            1.0]``.  Emission halts as soon as this value is exceeded.
        verbose: When ``True`` each emission attempt is printed to
            stdout in addition to being sent to the configured sink.
        prefix: Short label prepended to every log line, e.g.
            ``"warn"`` or ``"error"``.  Defaults to ``"info"``.
        retries: How many times to retry a failed emission before
            recording it as a permanent error.  Defaults to ``3``.
        jitter: Seconds of random delay added between retries to avoid
            thundering-herd bursts.  Defaults to ``0.1``.
        dry_run: When ``True`` the function performs all formatting and
            validation but does *not* contact the sink.  Defaults to
            ``False``.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ #
# Req 2a — Literal with string values only
# ------------------------------------------------------------------ #


def _example_string_literal(
    text: str,
    align: Literal["left", "center", "right"],
    mode: Literal["truncate", "wrap", "overflow"] = "wrap",
) -> None:
    """Render ``text`` inside a fixed-width column.

    Applies the requested text-alignment strategy and decides what to do
    when the content is wider than the available column width.

    Args:
        text: The string to render.  Newline characters are normalised to
            the platform line separator before processing.
        align: Horizontal alignment of the text within the column.  One
            of ``"left"``, ``"center"``, or ``"right"``.
        mode: Strategy applied when ``text`` exceeds the column width.
            ``"truncate"`` clips the content and appends an ellipsis;
            ``"wrap"`` soft-wraps at word boundaries; ``"overflow"``
            lets the content exceed the boundary without modification.
            Defaults to ``"wrap"``.

    Raises:
        ValueError: If ``text`` is empty after whitespace normalisation.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ #
# Req 2b — Literal with non-string values → NotImplementedError
# ------------------------------------------------------------------ #


def _example_int_literal(
    dataset: str,
    workers: Literal[1, 2, 4, 8],
) -> None:
    """Run a parallel processing job with a fixed number of workers.

    Args:
        dataset: Filesystem path or URI for the input data.
        workers: Number of worker threads. Must be one of 1, 2, 4, or 8.
    """
    raise NotImplementedError


def _example_mixed_literal(
    dataset: str,
    workers: Literal[1, 2, 4, 8],
    strategy: Literal[True, False, "auto"] = "auto",
) -> None:
    """Run a parallel processing job with a scheduling strategy.

    Args:
        dataset: Filesystem path or URI for the input data.
        workers: Number of worker threads. Must be one of 1, 2, 4, or 8.
        strategy: Scheduling strategy. Defaults to ``"auto"``.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ #
# Req 3 — Concrete Enum subclass parameter
# ------------------------------------------------------------------ #


def _example_enum_param(
    resource_id: str,
    severity: Severity,
    notify: bool = True,
    message: str = "",
) -> None:
    """Raise an alert for the specified resource.

    Creates an alert record in the monitoring backend and, when
    ``notify`` is ``True``, dispatches a notification through the
    configured channel (email, PagerDuty, etc.).

    This fixture ensures that a concrete ``Enum`` subclass (``Severity``)
    is converted to an appropriate ``enum`` schema entry rather than being
    treated as an opaque string or object.

    Args:
        resource_id: Unique identifier of the resource that triggered the
            alert, e.g. ``"server-42"`` or ``"pipeline/etl-prod"``.
        severity: Urgency classification for this alert.  Must be a
            ``Severity`` enum member (``LOW``, ``MEDIUM``, ``HIGH``, or
            ``CRITICAL``).
        notify: Whether to push an out-of-band notification in addition
            to writing the alert record.  Defaults to ``True``.
        message: Optional free-text annotation attached to the alert.
            Defaults to an empty string when not supplied.

    Raises:
        ValueError: If ``resource_id`` is an empty string.
        PermissionError: If the caller lacks write access to the alert
            backend.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ #
# Req 4 — List parameters: bare list + typed list[T]
# ------------------------------------------------------------------ #


def _example_list_params(
    tags: list,
    names: list[str],
    scores: list[int],
    weights: list[float] = [],  # noqa: B006 — intentional for schema fixture
    enabled_flags: list[bool] = [],  # noqa: B006
) -> None:
    """Filter and rank items by tag membership and numeric score.

    Accepts a heterogeneous ``tags`` list (untyped, to exercise the
    bare-``list`` case) alongside typed ``names``, ``scores``, and
    optional ``weights`` / ``enabled_flags`` lists.

    Args:
        tags: Arbitrary sequence of tag values used to pre-filter the
            candidate set.  May contain strings, integers, or any
            JSON-serialisable value; the caller is responsible for
            consistency with the stored tag schema.
        names: Ordered list of item identifiers to consider.  Each name
            must be a non-empty string.
        scores: Raw numeric score for each entry in ``names``.  Must
            have the same length as ``names``; values may be negative.
        weights: Optional per-name multipliers applied before ranking.
            When provided must match the length of ``names``.  Defaults
            to an empty list, in which case uniform weighting is used.
        enabled_flags: Boolean mask indicating which names are currently
            active.  When provided must match the length of ``names``.
            Defaults to an empty list, treated as all-enabled.

    Raises:
        ValueError: If ``names`` and ``scores`` differ in length, or if
            either ``weights`` or ``enabled_flags`` is non-empty and
            does not match the length of ``names``.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ #
# Req 5 — Dict parameters: bare dict + typed dict[str, T]
# ------------------------------------------------------------------ #


def _example_dict_params(
    metadata: dict,
    word_counts: dict[str, int],
    feature_flags: dict[str, bool],
    score_overrides: dict[str, float] = {},  # noqa: B006
) -> None:
    """Merge and normalise several dictionary inputs into a canonical record.

    Combines an untyped ``metadata`` blob with strictly typed
    ``word_counts`` and ``feature_flags`` mappings.  The untyped
    parameter tests that converters emit an ``object`` schema without
    ``additionalProperties`` constraints, while the typed parameters
    verify that ``additionalProperties`` is set to the correct primitive
    schema.

    Args:
        metadata: Free-form key/value pairs attached to the record.  All
            keys must be strings; values may be any JSON-serialisable
            type.  Unknown keys are preserved verbatim.
        word_counts: Mapping of token strings to their integer occurrence
            counts within the document.  All values must be non-negative.
        feature_flags: Mapping of feature-flag names to their enabled
            state.  A value of ``True`` means the flag is active for
            this record.
        score_overrides: Optional mapping of field names to floating-
            point score adjustments applied after the default ranking
            pass.  Defaults to an empty dict (no overrides).

    Raises:
        TypeError: If any value in ``word_counts`` is not an integer, or
            if any value in ``feature_flags`` is not a boolean.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ #
# Req 6 — Zero-argument callable
# ------------------------------------------------------------------ #


def _example_no_args() -> None:
    """Return the current health status of the service.

    Queries all registered subsystems (database, cache, message bus) and
    aggregates their individual health probes into a single summary
    object.  No arguments are accepted; the target endpoint is resolved
    from the module-level configuration at call time.

    Examples:
        >>> _example_no_args()
    """
    raise NotImplementedError


# ------------------------------------------------------------------ #
# Gap 1 — Optional[T] with None default → optional T schema, not anyOf
#
# Converter contract being asserted:
#   • Optional[T] + default None  → plain T schema, excluded from required
#   • plain T     + default None  → out of scope (not supported)
#   • plain T     + non-None default → plain T schema, excluded from required
#
# All three param styles are present so one test function can verify the
# full required/optional boundary in a single schema snapshot.
# ------------------------------------------------------------------ #


def _example_optional_args(  # noqa: PLR0913 — test fixture requires all 6 params to exercise required/optional boundary
    query: str,  # required str
    limit: int,  # required int
    filter_tag: str | None = None,  # Optional[str] + None → optional str
    min_score: float | None = None,  # Optional[float] + None → optional float
    include_deleted: bool | None = None,  # Optional[bool] + None → optional bool
    page_size: int = 20,  # plain int + non-None default → optional int
) -> None:
    """Search the index and return matching document identifiers.

    Executes a full-text query against the configured search backend.
    Optional parameters narrow the result set; when omitted they apply
    no additional constraint.

    The ``Optional[T]`` parameters in this fixture all carry ``None``
    as their default value.  The converter must treat each as an
    optional ``T`` field — not as a nullable / anyOf field — and must
    not include them in the ``required`` array.

    Args:
        query: Full-text search expression passed verbatim to the
            backend query parser.  Must be a non-empty string.
        limit: Maximum number of document identifiers to return.  Must
            be a positive integer.
        filter_tag: When provided, restricts results to documents that
            carry this exact tag string.  ``None`` applies no tag
            filter.
        min_score: Minimum relevance score threshold in the range
            ``[0.0, 1.0]``.  Documents scoring below this value are
            excluded.  ``None`` disables score filtering.
        include_deleted: When ``True``, soft-deleted documents are
            included in the result set.  When ``False`` or ``None``
            they are excluded.
        page_size: Number of results per internal pagination batch.
            Does not affect the total ``limit``; only controls backend
            chunking.  Defaults to ``20``.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ #
# Gap 3 — Nested containers
#
# Two parameters cover the two meaningful nesting shapes:
#   • list[list[str]]      — list whose elements are themselves lists
#   • dict[str, list[int]] — dict whose values are lists
#
# These are distinct from TypedDict nesting (Gap 8) because the schema
# emission path goes through ``items`` / ``additionalProperties``
# rather than ``properties`` + ``required``.
# ------------------------------------------------------------------ #


def _example_nested_containers(
    tag_groups: list[list[str]],  # list of lists
    index: dict[str, list[int]],  # dict with list values
    label_scores: dict[str, list[float]] = {},  # noqa: B006 — optional, dict with list values
) -> None:
    """Batch-index documents using grouped tags and position lists.

    Accepts pre-grouped tag strings and inverted-index position lists,
    merging them into the search backend's internal representation.

    Args:
        tag_groups: Collection of tag clusters, where each inner list
            holds the tag strings belonging to one cluster.  Clusters
            may overlap; duplicate tags within a cluster are ignored.
        index: Inverted index mapping each token string to the ordered
            list of integer byte-offsets at which it appears in the
            source document.  Offset lists must be sorted ascending.
        label_scores: Optional mapping of label strings to lists of
            per-occurrence confidence scores.  When provided, each
            score list must match the corresponding offset list length
            in ``index``.  Defaults to an empty dict.

    Raises:
        ValueError: If any offset list in ``index`` is not sorted in
            ascending order, or if a ``label_scores`` entry length
            does not match the corresponding ``index`` entry.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ #
# Gap 8 — TypedDict parameters: flat and one-level nested
#
# Two separate fixtures so a test failure pinpoints flat-expansion
# bugs independently from nesting/recursion bugs.
# ------------------------------------------------------------------ #


def _example_flat_typed_dict(
    image_path: str,
    crop: BoundingBox,  # flat TypedDict → inline object schema
    scale: float = 1.0,
) -> None:
    """Crop and optionally rescale a source image.

    Extracts the rectangular region described by ``crop`` from the image
    at ``image_path`` and applies a uniform scale factor.

    Args:
        image_path: Filesystem path or ``s3://`` URI to the source image.
            JPEG and PNG formats are supported.
        crop: Pixel-space rectangle defining the region to extract.  All
            four fields (``x``, ``y``, ``width``, ``height``) are
            required and must be non-negative integers.
        scale: Uniform scale factor applied to the cropped region before
            returning.  ``1.0`` returns the crop at native resolution;
            ``0.5`` halves both dimensions.  Defaults to ``1.0``.

    Raises:
        FileNotFoundError: If ``image_path`` does not resolve to a
            readable file or URI.
        ValueError: If any field of ``crop`` is negative, or if
            ``scale`` is not a positive number.
    """
    raise NotImplementedError


def _example_nested_typed_dict(
    image_path: str,
    hint: DetectionResult,  # TypedDict containing a TypedDict field
    min_confidence: float = 0.5,
    label_filter: str | None = None,  # Gap 1 + Gap 8 combined
) -> None:
    """Re-score a candidate detection against an updated model.

    Accepts a previously computed ``DetectionResult`` as a hint and
    runs a focused re-scoring pass restricted to the hinted bounding
    box, returning a refined confidence estimate.

    The ``hint`` parameter uses ``DetectionResult``, a ``TypedDict``
    that contains a nested ``BoundingBox`` field.  This exercises the
    converter's one-level TypedDict recursion: it must expand
    ``DetectionResult.box`` into a nested ``object`` schema with its
    own ``properties`` and ``required`` rather than emitting a bare
    ``object`` or raising.

    Args:
        image_path: Filesystem path or ``s3://`` URI to the source
            image.  Must be the same image that produced ``hint``.
        hint: Previously computed detection result used to seed the
            re-scoring pass.  The ``box`` field restricts the search
            area; ``label`` and ``confidence`` seed the prior
            distribution.
        min_confidence: Re-scored detections below this threshold are
            discarded.  Must be in the range ``[0.0, 1.0]``.  Defaults
            to ``0.5``.
        label_filter: When provided, only detections whose label exactly
            matches this string are considered.  ``None`` accepts any
            label.

    Raises:
        FileNotFoundError: If ``image_path`` does not resolve to a
            readable file or URI.
        ValueError: If ``min_confidence`` is outside ``[0.0, 1.0]``,
            or if any field of ``hint.box`` is negative.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ #
# Gap 5 — Degraded / missing docstrings
#
# Three degradation levels, each in its own callable so tests can be
# narrowly targeted.  All three are also collected into
# ``degraded_docstring_examples`` for a single parametrized smoke test.
#
#   _example_no_docstring      __doc__ is None at runtime; converter
#                              must emit parameter schemas from
#                              annotations alone, with empty descriptions.
#
#   _example_description_only  One-line summary, no Args section;
#                              converter must emit parameter schemas
#                              with empty per-argument descriptions.
#
#   _example_partial_args_doc  Args section present but intentionally
#                              omits ``offset``; converter must emit an
#                              empty description for the undocumented
#                              parameter — not raise, not skip it.
# ------------------------------------------------------------------ #


def _example_no_docstring(  # Gap 5a — __doc__ is None
    name: str,
    value: int,
    active: bool = True,
) -> None:
    raise NotImplementedError


def _example_description_only(  # Gap 5b — summary only, no Args
    name: str,
    value: int,
    active: bool = True,
) -> None:
    """Store a named integer value in the registry."""
    raise NotImplementedError


def _example_partial_args_doc(  # noqa: D417
    name: str,
    value: int,
    offset: int = 0,  # intentionally absent from Args section below
    active: bool = True,
) -> None:
    """Store a named integer value with an optional numeric adjustment.

    Writes ``value + offset`` to the registry under ``name``.  The
    entry is created if absent and overwritten if present.

    Args:
        name: Registry key under which the value is stored.  Must be a
            non-empty string containing only alphanumeric characters and
            underscores.
        value: Base integer to persist.  May be negative.
        active: When ``False`` the entry is written in a disabled state
            and excluded from aggregation queries.  Defaults to ``True``.

    Returns:
        The canonical key string used to store the entry, which may
        differ from ``name`` after normalisation.
    """
    # ``offset`` is intentionally absent from Args to exercise the
    # partial-docstring path — the converter must emit an empty
    # description for it rather than raising KeyError or dropping it.
    raise NotImplementedError


# ------------------------------------------------------------------ #
# pytest fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def example_primitives():
    """str/int/float/bool, mixed required + optional. Req 1."""
    return _example_primitives


@pytest.fixture
def example_string_literal():
    """String-only Literal. Req 2a."""
    return _example_string_literal


@pytest.fixture
def example_int_literal():
    """int-only Literal → NotImplementedError. Req 2b."""
    return _example_int_literal


@pytest.fixture
def example_mixed_literal():
    """Mixed Literal (int/bool/str) → NotImplementedError. Req 2b."""
    return _example_mixed_literal


@pytest.fixture
def example_enum_param():
    """Concrete Enum subclass parameter. Req 3."""
    return _example_enum_param


@pytest.fixture
def example_list_params():
    """Bare list + typed list[str/int/float/bool]. Req 4."""
    return _example_list_params


@pytest.fixture
def example_dict_params():
    """Bare dict + typed dict[str, int/bool/float]. Req 5."""
    return _example_dict_params


@pytest.fixture
def example_no_args():
    """Zero-argument callable. Req 6."""
    return _example_no_args


@pytest.fixture
def example_optional_args():
    """Optional[T] + None default → optional T schema. Gap 1."""
    return _example_optional_args


@pytest.fixture
def example_nested_containers():
    """list[list[str]] and dict[str, list[int]] parameters. Gap 3."""
    return _example_nested_containers


@pytest.fixture
def example_flat_typed_dict():
    """Flat TypedDict parameter (BoundingBox). Gap 8."""
    return _example_flat_typed_dict


@pytest.fixture
def example_nested_typed_dict():
    """TypedDict parameter with one nested TypedDict field. Gap 8."""
    return _example_nested_typed_dict


@pytest.fixture
def example_no_docstring():
    """Callable with no docstring at all. Gap 5a."""
    return _example_no_docstring


@pytest.fixture
def example_description_only():
    """Summary-only docstring, no Args section. Gap 5b."""
    return _example_description_only


@pytest.fixture
def example_partial_args_doc():
    """Args section that omits one parameter. Gap 5c."""
    return _example_partial_args_doc


@pytest.fixture
def degraded_docstring_examples():
    """All three degraded-docstring callables as a list.

    Intended for use with @pytest.mark.parametrize in a single
    smoke test that asserts the converter does not raise on any of
    them::

        @pytest.mark.parametrize("fn", degraded_docstring_examples())
        def test_converter_survives_bad_docs(fn):
            result = Tool.from_callable(fn)
            assert result is not None

    Note: pytest does not support injecting a fixture value directly
    into parametrize.  Use the raw list instead::

        @pytest.mark.parametrize("fn", [
            _example_no_docstring,
            _example_description_only,
            _example_partial_args_doc,
        ])
        def test_converter_survives_bad_docs(fn): ...
    """
    return [
        _example_no_docstring,
        _example_description_only,
        _example_partial_args_doc,
    ]


@pytest.fixture
def all_examples():
    """Every example callable as an ordered list.

    Suitable for smoke tests that verify the converter does not raise
    on any supported input shape.  Degraded-docstring callables are
    included deliberately — the assertion should be *no crash*, not a
    fully-populated schema.

    Same parametrize caveat as ``degraded_docstring_examples``: use
    the raw ``_example_*`` names in @pytest.mark.parametrize rather
    than this fixture.
    """
    return [
        _example_primitives,
        _example_string_literal,
        _example_enum_param,
        _example_list_params,
        _example_dict_params,
        _example_no_args,
        _example_optional_args,
        _example_nested_containers,
        _example_flat_typed_dict,
        _example_nested_typed_dict,
        _example_no_docstring,
        _example_description_only,
        _example_partial_args_doc,
    ]
