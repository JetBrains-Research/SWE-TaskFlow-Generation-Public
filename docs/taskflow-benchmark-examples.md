# TaskFlow benchmark examples

Choose an example:

1. [qutebrowser: encode search terms in URLs](#1-qutebrowser-encode-search-terms-in-urls)
2. [NodeBB: return HTTP 500 for failed uploads](#2-nodebb-return-http-500-for-failed-uploads)
3. [NodeBB: remove multiple list elements](#3-nodebb-remove-multiple-list-elements)

These examples show how a source issue becomes three incremental issue-solving
prompts and is paired with repository-understanding questions, reference
answers, and executable golden proofs. In the QA-first configuration, the QA
questions are presented before the three issue prompts.

## 1. qutebrowser: encode search terms in URLs

**QA-first dataset record:** [`qa_first_terra`, row 52](https://huggingface.co/datasets/JetBrains-Research/SWE-TaskFlow-SWE-Bench-Pro/viewer/default/qa_first_terra?row=52)

**Instance ID:** `instance_qutebrowser__qutebrowser-fec187c2cb53d769c2682b35ca77858a811414a8-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d`

### Issue

qutebrowser constructs search URLs from text entered by the user. Search terms
containing spaces or reserved characters must be safely encoded before they are
inserted into a URL.

For example:

```text
Input:  alpha/beta & gamma
Output: https://search.example/?q=alpha%2Fbeta%20%26%20gamma
```

Without correct encoding, the slash, ampersand, and spaces can change the
meaning or validity of the resulting URL.

### Three incremental issue prompts

**Prompt 1: identify the bug**

Fix search URL construction for terms containing spaces or special characters.

**Prompt 2: specify the encoding behavior**

Percent-encode search terms when inserting them into query parameters. Spaces
and special characters must be handled consistently.

**Prompt 3: generalize the behavior**

Preserve the same encoding behavior for search templates using different host
domains. No new public interface is required.

Each prompt adds information while the complete sequence preserves the original
task.

### Repository-understanding question

> How does configured search-engine input distinguish a recognized shortcut
> from an unknown leading word, and how is the resulting term carried into the
> selected search URL?

A shortcut is a configured short name for a search engine. For example, if
`w` identifies Wikipedia, `w quantum mechanics` selects Wikipedia and searches
for `quantum mechanics`.

### Reference answer

The configuration requires a `DEFAULT` search template and may define
additional named templates containing a `{}` placeholder. The input is stripped
and split once. Its first token is treated as a shortcut only if that token is a
configured key.

For a recognized shortcut, the shortcut is removed and its template is
selected. For an unknown leading word, the complete input remains the search
term and the `DEFAULT` template is selected. The term is then percent-encoded
with no safe characters, inserted into the selected template, and converted
into a `QUrl`.

### Golden proof

The proof runs the repository implementation with a controlled search-engine
configuration. It exercises both branches:

```text
Recognized shortcut
Input:  named alpha/beta & gamma
Engine: named
Term:   alpha/beta & gamma
URL:    https://named.invalid/find?term=alpha%2Fbeta%20%26%20gamma&source=qute

Unknown shortcut
Input:  unknown alpha/beta
Engine: DEFAULT
Term:   unknown alpha/beta
URL:    https://default.invalid/?term=unknown%20alpha%2Fbeta
```

The proof exits successfully, depends on the repository implementation, is
deterministic, and leaves the repository unchanged.

## 2. NodeBB: return HTTP 500 for failed uploads

**QA-first dataset record:** [`qa_first_terra`, row 53](https://huggingface.co/datasets/JetBrains-Research/SWE-TaskFlow-SWE-Bench-Pro/viewer/default/qa_first_terra?row=53)

**Instance ID:** `instance_nodebb__nodebb-8ca65b0c78c67c1653487c02d1135e1b702185e1-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e`

### Issue

The NodeBB admin endpoint for uploading a category image can report an error in
its JSON body while returning HTTP 200:

```http
HTTP/1.1 200 OK

{
  "error": "[[error:invalid-image-type, ...]]"
}
```

The body says that the request failed, but the status code says that it
succeeded. Browsers, AJAX clients, and monitoring systems can therefore treat a
failed upload as successful.

The expected response is:

```http
HTTP/1.1 500 Internal Server Error

{
  "error": "[[error:invalid-image-type, ...]]"
}
```

### Three incremental issue prompts

**Prompt 1: identify the protocol mismatch**

Failed admin uploads currently return HTTP 200 while placing the error only in
the JSON body. The affected endpoint is
`/api/admin/category/uploadpicture`.

**Prompt 2: fix server-side error propagation**

Make upload validation asynchronous and propagate unsupported MIME types and
malformed JSON as HTTP 500 responses. Apply the behavior consistently to the
category image, favicon, touch icon, and maskable icon upload paths.

**Prompt 3: make the client consume the error consistently**

Read API v3 messages, simple JSON error objects, and HTTP fallback text in a
defined order. Preserve the upload modal's formatting and existing successful
upload behavior.

### Repository-understanding question

> How does malformed JSON in a category-picture upload's `params` field travel
> from the upload controller to the API response? Include temporary-file
> handling, the handoff mechanism, the response status, and the response body
> shape for this non-v3 API route.

This question checks the existing error path rather than asking the agent to
repeat the issue requirements.

### Reference answer

The category-picture controller catches the `JSON.parse` failure, deletes the
temporary uploaded file, and calls `next` with
`Error("[[error:invalid-json]]")` instead of writing the response directly.

The application registers its URI and general error middleware after the normal
routes. Because `/api/admin/category/uploadpicture` is not an API v3 route, the
error reaches the general handler. That handler sets status 500 and returns an
API JSON object containing the request path, the `invalid-json` message, and the
computed `bodyClass`.

This also explains the original inconsistency: an unsupported MIME type used a
different path that wrote JSON directly without setting an error status.

### Golden proof

The proof invokes the real upload controller with malformed parameters and then
passes the controller-produced error into the repository's error middleware.
It observes:

```text
Temporary file: deleted
Forwarded error: [[error:invalid-json]]
HTTP status:    500
Response:
{
  "path": "/api/admin/category/uploadpicture",
  "error": "[[error:invalid-json]]",
  "bodyClass": "body-class"
}
```

The proof establishes the complete controller-to-middleware path at runtime
rather than merely searching the source code.

## 3. NodeBB: remove multiple list elements

**QA-first dataset record:** [`qa_first_terra`, row 46](https://huggingface.co/datasets/JetBrains-Research/SWE-TaskFlow-SWE-Bench-Pro/viewer/default/qa_first_terra?row=46)

**Instance ID:** `instance_nodebb__nodebb-397835a05a8e2897324e566b41c5e616e172b4af-v89631a1cdb318276acb48860c5d78077211397c6`

### Issue

The list-removal operation handles one element at a time but does not correctly
remove multiple distinct elements supplied in an array.

The reproduction is self-contained:

```text
Initial list: [a, b, c, d, e]
Remove:       [b, d]
Expected:     [a, c, e]
```

Every requested element that is present should be removed, and the relative
order of the remaining elements must be preserved.

### Three incremental issue prompts

**Prompt 1: add array removal**

Allow a single list-removal call to accept and remove multiple distinct
elements.

**Prompt 2: provide the concrete behavior**

Removing `[b, d]` from `[a, b, c, d, e]` must produce `[a, c, e]`.

**Prompt 3: define completion and scope**

Preserve the order of remaining values and ensure that a read performed after
the operation immediately observes the updated list. Validate that the key
exists and refers to a list.

### Repository-understanding questions

#### PostgreSQL prepend ordering

> How does a PostgreSQL array-prepend call progress from transaction setup
> through persistence so that supplying `[a, b, c]` places `c` first, and what
> does the empty-array path do?

#### Redis removal completion

> How does a Redis list removal call establish the point at which a subsequent
> full-range read sees the updated list, including the falsy-key path?

### Reference answers

#### PostgreSQL

`listPrepend` opens a transaction and verifies that the key has list type. It
then iterates through `a`, `b`, and `c` from left to right. Each database query
places its current value in front of the stored array:

```text
[] → [a] → [b, a] → [c, b, a]
```

An empty input still reaches the type check but executes no prepend queries.

#### Redis

`listRemoveAll` returns immediately for a falsy key. For a supplied key, it
awaits `client.lrem(key, 0, value)`. Its promise therefore cannot resolve before
Redis completes the removal.

A caller that awaits `listRemoveAll` and then calls `getListRange` reaches
`client.lrange` only after that completion boundary, so it observes the
post-removal list.

### Golden proofs

The PostgreSQL proof records the real adapter's query sequence and controlled
state transitions:

```text
Input:                  [a, b, c]
Stored result:          [c, b, a]
Prepend queries:        3
Additional queries for
an empty input:         0
```

The Redis proof delays the mocked `lrem` promise to make the completion boundary
observable:

```text
Before lrem resolves: removal promise has not completed
After lrem resolves:  range read returns [b, c]
Falsy key:            no additional Redis call
```

Both proofs execute repository code, produce deterministic observations, and
leave the repository unchanged.
