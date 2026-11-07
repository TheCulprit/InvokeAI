/**
 * A finished candidate keeps comparison active while its preview is hidden. For a selected placeholder, show its
 * live progress frame when available. All preview pixels land at the rect pinned at submission.
 */

import type { CanvasPlacementContract, Rect, StagedPreviewInput } from '@workbench/canvas-engine/api';

/** Inputs to {@link selectStagedPreviewSource}. */
export interface StagedPreviewSelection {
  /** Where the selected placeholder's result lands (its submitted rect); null when no placeholder is selected. */
  progressRect: Rect | null;
  /** The selected placeholder's denoise-progress frame, if any (b64 data URL + native dims). */
  progressImage: { dataUrl: string; width: number; height: number } | null;
  /** Whether finished candidate previews are visible. Placeholder progress is controlled separately. */
  isVisible: boolean;
  /** The selected staged candidate, if any, with its document-space placement. */
  selectedCandidate: { imageName: string; placement: CanvasPlacementContract } | null;
}

export const selectStagedPreviewSource = ({
  isVisible,
  progressImage,
  progressRect,
  selectedCandidate,
}: StagedPreviewSelection): StagedPreviewInput | null => {
  if (selectedCandidate) {
    return isVisible
      ? { imageName: selectedCandidate.imageName, placement: selectedCandidate.placement }
      : { hidden: true };
  }
  if (progressRect && progressImage && progressRect.width > 0 && progressRect.height > 0) {
    // Progress frames are low-res latents of the submitted region; scale to fill it.
    return { dataUrl: progressImage.dataUrl, placement: { ...progressRect, opacity: 1 } };
  }
  return null;
};

/** Key staged preview inputs so decoding reruns for source/progress changes only. */
export const stagedPreviewKey = (source: StagedPreviewInput | null): string => {
  if (source === null) {
    return 'none';
  }
  if ('hidden' in source) {
    return 'hidden';
  }
  const { placement } = source;
  const rect = `${placement.x}:${placement.y}:${placement.width}:${placement.height}:${placement.opacity}`;
  return 'imageName' in source ? `image:${source.imageName}:${rect}` : `data:${rect}:${source.dataUrl}`;
};
