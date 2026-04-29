from __future__ import annotations

from .models import AuthorVerificationResult, IncorrectVector, LayerInfo


def calculate_author_results(
    layer: LayerInfo,
    original_frames: tuple[int, ...],
    checked_vector_numbers: tuple[int, ...],
    incorrect_vectors: dict[int, IncorrectVector],
) -> tuple[AuthorVerificationResult, ...]:
    if len(original_frames) != len(checked_vector_numbers):
        raise ValueError("Количество исходных кадров не совпадает с количеством проверенных векторов.")

    checked_by_author: dict[str, list[int]] = {author: [] for author in layer.author_frames}
    incorrect_by_author: dict[str, list[int]] = {author: [] for author in layer.author_frames}

    for original_frame, vector_number in zip(original_frames, checked_vector_numbers, strict=True):
        vector = incorrect_vectors.get(vector_number)
        for author, author_frames in layer.author_frames.items():
            if original_frame not in author_frames:
                continue
            checked_by_author[author].append(original_frame)
            if vector is not None and not vector.is_correct:
                incorrect_by_author[author].append(original_frame)

    return tuple(
        AuthorVerificationResult(
            author=author,
            checked_frames=tuple(checked_by_author[author]),
            incorrect_frames=tuple(incorrect_by_author[author]),
        )
        for author in layer.author_frames
    )
