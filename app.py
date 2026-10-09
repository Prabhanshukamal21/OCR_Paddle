import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from utils.image_utils import load_uploaded_image
from utils.ocr_engine import extract_text
from utils.pdf_utils import pdf_to_images
from utils.chunking import chunk_text

from services.embedding_service import (
    EmbeddingService
)

from services.chroma_service import (
    ChromaService
)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Bilingual OCR + ChromaDB",
    page_icon=" ",
    layout="wide"
)


st.title(
    "OCR System"
)

st.caption(
    "OCR → "
    "Embeddings → ChromaDB"
)


# --------------------------------------------------
# SERVICES
# --------------------------------------------------

@st.cache_resource
def get_services():

    embedding_service = (
        EmbeddingService()
    )

    chroma_service = ChromaService(
        persist_directory="data/chroma_db",

        # New collection because the embedding
        # model has changed.
        collection_name=(
            "ocr_documents_bilingual"
        ),

        embedding_service=embedding_service
    )

    return (
        embedding_service,
        chroma_service
    )


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab_ocr, tab_search, tab_records = st.tabs(
    [
        "Extract & Store",
        "Semantic Search",
        "Stored Records"
    ]
)


# ==================================================
# TAB 1
# ==================================================

with tab_ocr:

    st.subheader(
        "Upload Image or PDF"
    )

    uploaded_file = st.file_uploader(
        "Upload an image or PDF",

        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "webp",
            "pdf"
        ],

        help=(
            "Supports English + Hindi "
            "text."
        )
    )

    if uploaded_file:

        file_name = (
            uploaded_file.name
        )

        file_bytes = (
            uploaded_file.getvalue()
        )

        file_extension = (
            Path(file_name)
            .suffix
            .lower()
        )

        # --------------------------------------------------
        # IMAGE
        # --------------------------------------------------

        if file_extension != ".pdf":

            try:

                image = load_uploaded_image(
                    file_bytes
                )

                st.image(
                    image,
                    caption=(
                        f"{file_name} · "
                        f"{image.width} × "
                        f"{image.height}"
                    ),
                    use_container_width=True
                )

                if st.button(
                    "Run OCR",
                    type="primary"
                ):

                    with st.spinner(
                        "Recognizing text..."
                    ):

                        result = extract_text(
                            image
                        )

                    st.session_state[
                        "latest_ocr_text"
                    ] = result["text"]

                    st.session_state[
                        "latest_ocr_details"
                    ] = result

                    st.session_state[
                        "latest_ocr_filename"
                    ] = file_name

                    st.success(
                        "OCR completed."
                    )

            except Exception as exc:

                st.error(
                    f"Could not process image: {exc}"
                )

        # --------------------------------------------------
        # PDF
        # --------------------------------------------------

        else:

            st.info(
                "PDF detected. "
                "Each page will be converted to an image "
                "and processed with bilingual OCR."
            )

            if st.button(
                "Run OCR",
                type="primary"
            ):

                try:

                    with st.spinner(
                        "Converting PDF pages..."
                    ):

                        pages = pdf_to_images(
                            file_bytes
                        )

                    if not pages:

                        st.warning(
                            "The PDF contains no pages."
                        )

                    else:

                        all_page_text = []
                        all_scores = []

                        progress = st.progress(
                            0
                        )

                        for index, page_data in enumerate(
                            pages
                        ):

                            page_number = (
                                page_data["page"]
                            )

                            page_image = (
                                page_data["image"]
                            )

                            with st.expander(
                                f"Page {page_number}"
                            ):

                                st.image(
                                    page_image,
                                    caption=(
                                        f"PDF Page "
                                        f"{page_number}"
                                    ),
                                    use_container_width=True
                                )

                                page_result = (
                                    extract_text(
                                        page_image
                                    )
                                )

                                page_text = (
                                    page_result["text"]
                                )

                                if page_text.strip():

                                    all_page_text.append(
                                        f"[Page {page_number}]\n"
                                        f"{page_text}"
                                    )

                                all_scores.extend(
                                    page_result.get(
                                        "scores",
                                        []
                                    )
                                )

                                st.text_area(
                                    "Recognized text",
                                    value=page_text,
                                    height=180,
                                    key=(
                                        f"page_text_"
                                        f"{page_number}"
                                    )
                                )

                            progress.progress(
                                (index + 1)
                                / len(pages)
                            )

                        combined_text = (
                            "\n\n".join(
                                all_page_text
                            )
                        )

                        mean_confidence = None

                        if all_scores:

                            mean_confidence = (
                                sum(all_scores)
                                / len(all_scores)
                            )

                        result = {
                            "text": combined_text,
                            "lines": combined_text.splitlines(),
                            "scores": all_scores,
                            "mean_confidence": (
                                mean_confidence
                            )
                        }

                        st.session_state[
                            "latest_ocr_text"
                        ] = combined_text

                        st.session_state[
                            "latest_ocr_details"
                        ] = result

                        st.session_state[
                            "latest_ocr_filename"
                        ] = file_name

                        st.success(
                            f"OCR completed for "
                            f"{len(pages)} page(s)."
                        )

                except Exception as exc:

                    st.error(
                        f"Could not process PDF: {exc}"
                    )


    # ==================================================
    # OCR RESULT
    # ==================================================

    if st.session_state.get(
        "latest_ocr_text"
    ):

        st.divider()

        st.subheader(
            "Extracted Text"
        )

        extracted_text = st.text_area(
            "English + Hindi text "
            "(you can edit it before storing)",

            value=st.session_state.get(
                "latest_ocr_text",
                ""
            ),

            height=300,

            key="editable_ocr_text"
        )

        # --------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------

        details = st.session_state.get(
            "latest_ocr_details",
            {}
        )

        confidence = details.get(
            "mean_confidence"
        )

        if confidence is not None:

            st.metric(
                "Average OCR Confidence",
                f"{confidence:.2%}"
            )

        # --------------------------------------------------
        # DOWNLOAD
        # --------------------------------------------------

        output_name = Path(
            st.session_state.get(
                "latest_ocr_filename",
                "ocr_output"
            )
        ).stem

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "⬇Download TXT",

                data=extracted_text,

                file_name=(
                    f"{output_name}.txt"
                ),

                mime="text/plain"
            )

        with col2:

            json_data = json.dumps(
                {
                    "source": st.session_state.get(
                        "latest_ocr_filename",
                        "unknown"
                    ),

                    "text": extracted_text,

                    "created_at": (
                        datetime.now()
                        .isoformat(
                            timespec="seconds"
                        )
                    )
                },

                ensure_ascii=False,

                indent=2
            )

            st.download_button(
                "⬇Download JSON",

                data=json_data,

                file_name=(
                    f"{output_name}.json"
                ),

                mime="application/json"
            )

        # --------------------------------------------------
        # STORE
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "Store in ChromaDB"
        )

        chunk_size = st.slider(
            "Chunk size",
            min_value=200,
            max_value=1000,
            value=500,
            step=50
        )

        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=300,
            value=100,
            step=25
        )

        if st.button(
            "Create Embeddings & Store",
            type="primary"
        ):

            try:

                chunks = chunk_text(
                    extracted_text,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap
                )

                if not chunks:

                    st.warning(
                        "There is no text to store."
                    )

                else:

                    (
                        embedding_service,
                        chroma_service
                    ) = get_services()

                    result = (
                        chroma_service
                        .add_chunks(
                            chunks=chunks,

                            source=(
                                st.session_state.get(
                                    "latest_ocr_filename",
                                    "uploaded_file"
                                )
                            ),

                            base_metadata={
                                "source_type": (
                                    "pdf_ocr"
                                    if file_extension == ".pdf"
                                    else "image_ocr"
                                ),

                                "created_at": (
                                    datetime.now()
                                    .isoformat(
                                        timespec="seconds"
                                    )
                                ),

                                "language": (
                                    "English + Hindi"
                                )
                            }
                        )
                    )

                    st.success(
                        f"Stored "
                        f"{result['count']} chunk(s) "
                        f"in ChromaDB."
                    )

                    st.info(
                        "Embedding dimensions: "
                        f"{result['embedding_dimensions']}"
                    )

            except Exception as exc:

                st.error(
                    f"Could not store text: {exc}"
                )


# ==================================================
# TAB 2
# ==================================================

with tab_search:

    st.subheader(
        "Multilingual Semantic Search"
    )

    st.write(
        "Search using English or Hindi. "
        "The query is converted into an embedding "
        "and compared with stored OCR text."
    )

    query = st.text_input(
        "Search query",

        placeholder=(
            "Example: OCR system "
            "or "
            "ओसीआर सिस्टम"
        )
    )

    n_results = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button(
        "Search",
        type="primary"
    ):

        if not query.strip():

            st.warning(
                "Enter a search query."
            )

        else:

            try:

                (
                    embedding_service,
                    chroma_service
                ) = get_services()

                matches = (
                    chroma_service.search(
                        query=query.strip(),
                        n_results=n_results
                    )
                )

                if not matches:

                    st.info(
                        "No stored records found."
                    )

                else:

                    st.success(
                        f"Found "
                        f"{len(matches)} result(s)."
                    )

                    for index, match in enumerate(
                        matches,
                        start=1
                    ):

                        st.markdown(
                            f"### Result {index}"
                        )

                        st.write(
                            match["text"]
                        )

                        metadata = (
                            match["metadata"]
                        )

                        st.caption(
                            f"Source: "
                            f"{match['source']} | "
                            f"Distance: "
                            f"{match['distance']}"
                        )

                        if metadata:

                            st.json(
                                metadata
                            )

                        st.divider()

            except Exception as exc:

                st.error(
                    f"Search failed: {exc}"
                )


# ==================================================
# TAB 3
# ==================================================

with tab_records:

    st.subheader(
        "🗂️ Stored ChromaDB Records"
    )

    if st.button(
        "Refresh Records"
    ):

        st.cache_resource.clear()

        st.rerun()

    try:

        (
            embedding_service,
            chroma_service
        ) = get_services()

        records = (
            chroma_service
            .list_records(
                limit=100
            )
        )

        if not records:

            st.info(
                "No records stored yet."
            )

        else:

            st.write(
                f"Total records: "
                f"{len(records)}"
            )

            for index, record in enumerate(
                records,
                start=1
            ):

                with st.expander(
                    f"Record {index} · "
                    f"{record['source']}"
                ):

                    st.write(
                        record["text"]
                    )

                    st.json(
                        record["metadata"]
                    )

    except Exception as exc:

        st.error(
            f"Could not load records: {exc}"
        )