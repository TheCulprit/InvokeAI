import { useDndContext } from '@dnd-kit/core';
import { Box, Spacer, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useScopeOnFocus } from 'common/hooks/interactionScopes';
import { CanvasPanelContent } from 'features/controlLayers/components/CanvasPanelContent';
import { CanvasSendToToggle } from 'features/controlLayers/components/CanvasSendToToggle';
import { selectSendToCanvas } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectEntityCountActive } from 'features/controlLayers/store/selectors';
import GalleryPanelContent from 'features/gallery/components/GalleryPanelContent';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasRightPanelContent = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  const [tab, setTab] = useState(0);
  useScopeOnFocus('gallery', ref);

  return (
    <Tabs index={tab} onChange={setTab} w="full" h="full" display="flex" flexDir="column">
      <TabList alignItems="center">
        <PanelTabs setTab={setTab} />
        <Spacer />
        <CanvasSendToToggle />
      </TabList>
      <TabPanels w="full" h="full">
        <TabPanel w="full" h="full" p={0} pt={3}>
          <CanvasPanelContent />
        </TabPanel>
        <TabPanel w="full" h="full" p={0} pt={3}>
          <GalleryPanelContent />
        </TabPanel>
      </TabPanels>
    </Tabs>
  );
});

CanvasRightPanelContent.displayName = 'CanvasRightPanelContent';

const PanelTabs = memo(({ setTab }: { setTab: (val: number) => void }) => {
  const { t } = useTranslation();
  const activeEntityCount = useAppSelector(selectEntityCountActive);
  const sendToCanvas = useAppSelector(selectSendToCanvas);
  const tabTimeout = useRef<number | null>(null);
  const dndCtx = useDndContext();

  const onOnMouseOverLayersTab = useCallback(() => {
    tabTimeout.current = window.setTimeout(() => {
      if (dndCtx.active) {
        setTab(0);
      }
    }, 300);
  }, [dndCtx.active, setTab]);

  const onOnMouseOverGalleryTab = useCallback(() => {
    tabTimeout.current = window.setTimeout(() => {
      if (dndCtx.active) {
        setTab(1);
      }
    }, 300);
  }, [dndCtx.active, setTab]);

  const onMouseOut = useCallback(() => {
    if (tabTimeout.current) {
      clearTimeout(tabTimeout.current);
    }
  }, []);

  const layersTabLabel = useMemo(() => {
    if (activeEntityCount === 0) {
      return t('controlLayers.layer_other');
    }
    return `${t('controlLayers.layer_other')} (${activeEntityCount})`;
  }, [activeEntityCount, t]);

  return (
    <>
      <Tab position="relative" onMouseOver={onOnMouseOverLayersTab} onMouseOut={onMouseOut} w={32}>
        <Box as="span" w="full">
          {layersTabLabel}
        </Box>
        {sendToCanvas && (
          <Box position="absolute" top={2} right={2} h={2} w={2} bg="invokeYellow.300" borderRadius="full" />
        )}
      </Tab>
      <Tab position="relative" onMouseOver={onOnMouseOverGalleryTab} onMouseOut={onMouseOut}>
        {t('gallery.gallery')}
        {!sendToCanvas && (
          <Box position="absolute" top={2} right={2} h={2} w={2} bg="invokeYellow.300" borderRadius="full" />
        )}
      </Tab>
    </>
  );
});

PanelTabs.displayName = 'PanelTabs';
