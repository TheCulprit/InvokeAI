import type { SystemStyleObject } from '@chakra-ui/react';
import type { ViewerAutoSwitchMode } from '@workbench/settings/contracts';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import {
  ActivityIcon,
  CheckIcon,
  GalleryHorizontalEndIcon,
  LocateOffIcon,
  LockIcon,
  LockOpenIcon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const MODES: readonly ViewerAutoSwitchMode[] = ['off', 'progress', 'latest'];

/** Both widgets name the shared modes with the staging bar's keys, so existing translations carry over. */
const MODE_LABEL_KEYS: Record<ViewerAutoSwitchMode, string> = {
  latest: 'widgets.canvas.staging.autoSwitchLatest',
  off: 'widgets.canvas.staging.autoSwitchOff',
  progress: 'widgets.canvas.staging.autoSwitchProgress',
};

/** Icons by meaning; Pin is not Off because the filmstrip already marks pinned sessions with it. */
const MODE_ICONS: Record<ViewerAutoSwitchMode, LucideIcon> = {
  latest: GalleryHorizontalEndIcon,
  off: LocateOffIcon,
  progress: ActivityIcon,
};

/** Trigger styles per host: a widget header's ghost buttons or a floating toolbar's working size. */
const TRIGGER_CHROME = {
  header: { button: { color: 'fg.muted', size: 'sm' } },
  toolbar: { button: { size: 'md' } },
} as const;

type AutoSwitchMenuProps = {
  value: ViewerAutoSwitchMode;
  onChange: (mode: ViewerAutoSwitchMode) => void;
  locked: boolean;
  onLockedChange: (locked: boolean) => void;
  positioning: { placement: 'bottom-end' | 'top-end' };
  /** `full` shows the current mode beside the icon; `compact` is the icon alone, named by its tooltip. */
  density: 'compact' | 'full';
  /** Which neighbouring controls the trigger matches; `toolbar` by default. */
  chrome?: keyof typeof TRIGGER_CHROME;
  /** A host's container rules for the trigger and its mode text, such as the staging bar's narrow collapse. */
  triggerCss?: SystemStyleObject;
  valueCss?: SystemStyleObject;
};

/** What switches the view to new work: a menu of the modes, with the current one checked. */
export const AutoSwitchMenu = ({
  chrome = 'toolbar',
  density,
  positioning,
  triggerCss,
  value,
  valueCss,
  locked,
  onLockedChange,
  onChange,
}: AutoSwitchMenuProps) => {
  const { t } = useTranslation();
  const ids = useTooltipTriggerIds();
  const title = t('widgets.canvas.staging.autoSwitch');
  // Names the current mode on the trigger and its tooltip, so a compact, icon-only trigger still says it.
  const currentLabel = t('widgets.canvas.staging.autoSwitchCurrent', { mode: t(MODE_LABEL_KEYS[value]) });
  const ModeIcon = MODE_ICONS[value];
  const handleValueChange = useCallback(
    ({ value }: { value: string }) => {
      const mode = MODES.find((candidate) => candidate === value);

      if (mode) {
        onChange(mode);
      }
    },
    [onChange]
  );

  const toggleLocked = useCallback(() => onLockedChange(!locked), [locked, onLockedChange]);
  const lockLabel = t(locked ? 'widgets.canvas.staging.unlockAutoSwitch' : 'widgets.canvas.staging.lockAutoSwitch');

  return (
    <HStack bg="bg.subtle" borderColor="border" borderWidth="1px" flexShrink={0} gap="0" rounded="control">
      <Menu.Root ids={ids} positioning={positioning}>
        <Tooltip content={currentLabel} ids={ids}>
          <Menu.Trigger asChild>
            <Button
              aria-label={currentLabel}
              borderEndRadius="0"
              css={triggerCss}
              // Icon alone, it is square like the icon buttons beside it.
              px={density === 'compact' ? '0' : undefined}
              variant="ghost"
              {...TRIGGER_CHROME[chrome].button}
            >
              <ModeIcon aria-hidden />
              {density === 'full' ? (
                <Text css={valueCss} fontSize="md">
                  {t(MODE_LABEL_KEYS[value])}
                </Text>
              ) : null}
            </Button>
          </Menu.Trigger>
        </Tooltip>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="8rem" py="1">
              <Menu.RadioItemGroup value={value} onValueChange={handleValueChange}>
                <Menu.ItemGroupLabel>{title}</Menu.ItemGroupLabel>
                {MODES.map((mode) => (
                  <Menu.RadioItem key={mode} value={mode}>
                    <Menu.ItemIndicator>
                      <CheckIcon size={12} />
                    </Menu.ItemIndicator>
                    <Icon aria-hidden as={MODE_ICONS[mode]} boxSize="3.5" color="fg.subtle" flexShrink={0} />
                    <Menu.ItemText fontSize="md">{t(MODE_LABEL_KEYS[mode])}</Menu.ItemText>
                  </Menu.RadioItem>
                ))}
              </Menu.RadioItemGroup>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <IconButton
        {...TRIGGER_CHROME[chrome].button}
        aria-label={lockLabel}
        aria-pressed={locked}
        borderInlineStartColor="border"
        borderInlineStartWidth="1px"
        borderStartRadius="0"
        color={locked ? 'accent.solid' : 'fg.muted'}
        tooltip={lockLabel}
        variant="ghost"
        onClick={toggleLocked}
      >
        {locked ? <LockIcon aria-hidden /> : <LockOpenIcon aria-hidden />}
      </IconButton>
    </HStack>
  );
};
