/* oxlint-disable react-perf/jsx-no-jsx-as-prop -- Tooltip takes a content slot, which the React Compiler memoizes. */
import type { ReactElement } from 'react';

import { Box, Image } from '@chakra-ui/react';
import { MODEL_PREVIEW_WIDTH } from '@theme/scale';

import { Tooltip, type TooltipProps } from './Tooltip';

const CONTENT_PROPS = {
  bg: 'bg.panel',
  borderColor: 'border',
  borderWidth: '1px',
  maxW: 'none',
  p: '1',
  pointerEvents: 'none',
} satisfies TooltipProps['contentProps'];

const PREVIEW_WIDTH = `min(${MODEL_PREVIEW_WIDTH}, 70vw)`;
const PREVIEW_MAX_HEIGHT = `min(calc(${MODEL_PREVIEW_WIDTH} * 1.25), 60vh)`;

export const ImagePreviewTooltip = ({ children, src }: { children: ReactElement; src: string }) => {
  const preview = (
    <Image alt="" h="auto" maxH={PREVIEW_MAX_HEIGHT} objectFit="contain" rounded="sm" src={src} w={PREVIEW_WIDTH} />
  );
  return (
    <Tooltip
      closeDelay={0}
      content={preview}
      contentProps={CONTENT_PROPS}
      lazyMount
      openDelay={220}
      placement="bottom-start"
      showArrow={false}
      unmountOnExit
    >
      {/* Keep the tooltip's trigger state off the thumbnail's image and avatar elements. */}
      <Box as="span" display="inline-flex" flexShrink={0}>
        {children}
      </Box>
    </Tooltip>
  );
};
