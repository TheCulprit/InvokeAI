/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop -- ListItem slots take JSX; the React Compiler memoizes them. */
import type { ConceptUpdate } from '@features/generation/core/conceptList';
import type { GenerateLora, LoraModelConfig } from '@features/generation/core/types';
import type { MouseEvent, ReactNode } from 'react';

import { Avatar, Badge, Box, chakra, Flex, Icon, Menu, Portal } from '@chakra-ui/react';
import { DEFAULT_LORA_WEIGHT_CONFIG, getDefaultLoraWeight } from '@features/generation/core/settings';
import { useOptionalGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { useDebouncedDraftValue } from '@features/generation/ui/useDebouncedDraftValue';
import { useRegisterDraftFlusher } from '@platform/react/draftRegistry';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { IconButton } from '@platform/ui/Button';
import { ImagePreviewTooltip } from '@platform/ui/ImagePreviewTooltip';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListStack } from '@platform/ui/list/ListStack';
import { MenuActionItem, MenuContent, useContextMenu } from '@platform/ui/Menu';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { Tooltip } from '@platform/ui/Tooltip';
import {
  BoxIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ExternalLinkIcon,
  PowerIcon,
  PowerOffIcon,
  RotateCcwIcon,
  Trash2Icon,
} from 'lucide-react';
import { createContext, memo, use, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { GenerateFieldContextMenu } from './GenerateFieldContextMenu';
import { GenerateRowSwitch } from './GenerateToggleSwitch';

const WEIGHT_MARKS = [-1, 0, 1, 2];
const FALLBACK_MODELS: ConceptModelPort = {
  getBaseColorPalette: () => 'gray',
  getBaseLabel: (base) => base,
  getImageUrl: () => '',
};

/** The model catalog's helpers a concept row needs; the caller supplies them through its own UI port. */
export interface ConceptModelPort {
  getBaseColorPalette(base: string): string;
  getBaseLabel(base: string): string;
  getImageUrl(key: string, cacheKey?: string, size?: 'thumbnail' | 'preview'): string;
  /** Absent when this session may not manage models; the row then offers no manager link. */
  openInModelManager?: (key: string) => void;
}

export interface ConceptRowProps {
  /** False dims the row, badges it, and locks it off; the caller decides compatibility against its main model. */
  isCompatible?: boolean;
  /**
   * False badges the row as not installed, an error that blocks Invoke while the concept is on: its switch keeps the
   * saved state but is locked, and `issue` says what blocks and how to fix it.
   */
  isInstalled?: boolean;
  /** Shown beside the switch: the issue that blocks Invoke because of this concept, with its fix (an icon button). */
  issue?: ReactNode;
  /** The committed concept. Weight edits stay in a local draft until idle or a workbench flush. */
  lora: GenerateLora;
  models: ConceptModelPort;
  onRemove: (key: string) => void;
  /** Weight commits are debounced; toggles commit immediately. Apply updates to the latest settings. */
  onUpdate: (key: string, update: ConceptUpdate) => void;
}

/** A removed row unmounts with keyboard focus inside it; the next (else previous) row's button takes it instead. */
const findNeighbourRow = (from: Element | null): HTMLElement | null => {
  const item = from?.closest('[role="listitem"]');

  for (const step of ['nextElementSibling', 'previousElementSibling'] as const) {
    for (let sibling = item?.[step]; sibling; sibling = sibling[step]) {
      const primary = sibling.querySelector<HTMLElement>('[data-list-primary]');

      if (primary) {
        return primary;
      }
    }
  }

  return null;
};

interface ListSlot {
  container: Element;
  next: Element | null;
}

/** Where the list sat at each ancestor level, so focus can be placed after the last row unmounts it. */
const captureListSlots = (from: Element | null): ListSlot[] => {
  const slots: ListSlot[] = [];

  for (let node = from?.closest('[role="list"]'); node?.parentElement; node = node.parentElement) {
    slots.push({ container: node.parentElement, next: node.nextElementSibling });
  }

  return slots;
};

const TABBABLE = 'a[href], button, input:not([type="hidden"]), select, textarea, [tabindex], [contenteditable="true"]';

/**
 * The control before the removed list, read after the removal renders: the concept picker, which may only now be
 * enabled once the removed concept is no longer excluded from it.
 */
const findControlBeforeList = (slots: ListSlot[]): HTMLElement | null => {
  for (const { container, next } of slots) {
    if (!container.isConnected) {
      continue;
    }

    const preceding = [...container.querySelectorAll<HTMLElement>(TABBABLE)].filter(
      (element) =>
        element.tabIndex >= 0 &&
        !element.matches(':disabled') &&
        element.getClientRects().length > 0 &&
        (!next?.isConnected || Boolean(next.compareDocumentPosition(element) & Node.DOCUMENT_POSITION_PRECEDING))
    );

    if (preceding.length > 0) {
      return preceding.at(-1) ?? null;
    }
  }

  return null;
};

const ConceptProjectContext = createContext<string | undefined>(undefined);

/** A short concept list shares its project lifetime with the rows' drafts and menus. */
export const ConceptList = ({
  children,
  label,
  projectId,
}: {
  children: ReactNode;
  label: string;
  projectId: string;
}) => (
  <ConceptProjectContext value={projectId}>
    <ListStack dividers label={label}>
      {children}
    </ListStack>
  </ConceptProjectContext>
);

/**
 * One applied LoRA/concept: a list row (thumbnail, identity, toggle, remove, and a context menu) with its weight
 * scrubber as the row's detail, so the whole item shares one hover surface.
 */
export const ConceptRow = memo(function ConceptRow(props: ConceptRowProps) {
  const projectId = use(ConceptProjectContext);
  const { lora, onUpdate } = props;
  const commitWeight = useCallback(
    (weight: number) => onUpdate(lora.model.key, { weight }),
    [lora.model.key, onUpdate]
  );
  const {
    draftValue: weight,
    flushDraftValue,
    setDraftValue,
  } = useDebouncedDraftValue({
    delayMs: 250,
    onCommit: commitWeight,
    resetKey: projectId,
    value: lora.weight,
  });
  useRegisterDraftFlusher(flushDraftValue);
  const update = useCallback(
    (key: string, update: ConceptUpdate) => {
      if (update.weight === undefined) {
        onUpdate(key, update);
      } else {
        setDraftValue(update.weight);
      }
    },
    [onUpdate, setDraftValue]
  );

  // Reset the menu on project changes without remounting/flushing the weight draft into the next project.
  return <ConceptRowContent key={projectId} {...props} weight={weight} onUpdate={update} />;
});

/** Workflow weights preview locally, then commit one graph edit for the complete gesture. */
export const WorkflowConceptRow = ({
  compact,
  isEnabled,
  model,
  onRemove,
  onUpdate,
  weight,
  weightError,
}: {
  compact: boolean;
  isEnabled: boolean;
  model: LoraModelConfig;
  onRemove: () => void;
  onUpdate: (update: ConceptUpdate) => void;
  weight: number | null;
  weightError?: string;
}) => {
  const models = useOptionalGenerationUi()?.models;
  const installed = models?.catalog.find((item) => item.type === 'lora' && item.key === model.key);
  const resolved = installed ? { ...installed, type: 'lora' as const } : model;
  const [preview, setPreview] = useState<number | null>(null);
  const value = preview ?? weight ?? getDefaultLoraWeight(resolved);
  const update = useCallback((_key: string, patch: ConceptUpdate) => onUpdate(patch), [onUpdate]);
  const commitWeight = useCallback(
    (next: number) => {
      setPreview(null);
      if (next !== weight) {
        onUpdate({ weight: next });
      }
    },
    [onUpdate, weight]
  );

  return (
    <ConceptRowContent
      compact={compact}
      lora={{ isEnabled, model: resolved, weight: value }}
      models={models ?? FALLBACK_MODELS}
      weight={value}
      weightError={weightError}
      onPreviewWeight={setPreview}
      onRemove={onRemove}
      onUpdate={update}
      onWeightChangeEnd={commitWeight}
    />
  );
};

const ConceptRowContent = ({
  compact = false,
  isCompatible: isCompatibleModel = true,
  isInstalled = true,
  issue,
  lora,
  models,
  onRemove,
  onUpdate,
  weight,
  weightError,
  onPreviewWeight,
  onWeightChangeEnd,
}: ConceptRowProps & {
  compact?: boolean;
  weight: number;
  weightError?: string;
  onPreviewWeight?: (weight: number) => void;
  onWeightChangeEnd?: (weight: number) => void;
}) => {
  const { t } = useTranslation();
  const { key, name, trigger_phrases: triggerPhrases } = lora.model;
  const coverUrl = lora.model.cover_image ? models.getImageUrl(key) : null;
  const thumbnail = (
    <Avatar.Root
      bg="bg.muted"
      borderColor="border"
      borderWidth="1px"
      color="fg.subtle"
      flexShrink={0}
      shape="rounded"
      size={compact ? 'lg' : 'xl'}
    >
      <Avatar.Fallback>
        <Icon as={BoxIcon} boxSize="4" />
      </Avatar.Fallback>
      {coverUrl ? <Avatar.Image alt="" src={coverUrl} /> : null}
    </Avatar.Root>
  );
  // A concept that is not installed cannot run either; the row says which of the two it is.
  const isCompatible = isCompatibleModel && isInstalled;
  const isActive = lora.isEnabled && isCompatible;
  const defaultWeight = getDefaultLoraWeight(lora.model);
  const menu = useContextMenu();
  // Kept through the exit animation: the anchor clears the moment the menu closes.
  const { release: releaseMenu, value: shownAnchor } = useExitRetainedValue(menu.anchor);
  const handleToggle = useCallback((isEnabled: boolean) => onUpdate(key, { isEnabled }), [key, onUpdate]);
  const handleWeightChange = useCallback((weight: number) => onUpdate(key, { weight }), [key, onUpdate]);
  const stepWeight = useCallback(
    (direction: number) => {
      const { coarseStep, numberInputMin, numberInputMax } = DEFAULT_LORA_WEIGHT_CONFIG;
      const next = Math.round((weight + coarseStep * direction) * 1e10) / 1e10;
      onWeightChangeEnd?.(Math.min(numberInputMax, Math.max(numberInputMin, next)));
    },
    [onWeightChangeEnd, weight]
  );
  const handleIncrease = useCallback(() => stepWeight(1), [stepWeight]);
  const handleDecrease = useCallback(() => stepWeight(-1), [stepWeight]);
  const removeFrom = useCallback(
    (origin: Element | null) => {
      const neighbour = findNeighbourRow(origin);
      const slots = neighbour ? [] : captureListSlots(origin);

      onRemove(key);
      requestAnimationFrame(() => (neighbour ?? findControlBeforeList(slots))?.focus());
    },
    [key, onRemove]
  );
  const handleRemoveClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => removeFrom(event.currentTarget),
    [removeFrom]
  );
  const handleReset = useCallback(() => onUpdate(key, { weight: defaultWeight }), [defaultWeight, key, onUpdate]);
  const copyWeight = useCallback(() => String(weight), [weight]);
  const { openInModelManager } = models;

  return (
    <Box role="listitem">
      {/* Naming the group gives the row's "Weight" slider and controls their concept's context. */}
      {/* Only a concept the model skips is dimmed; one that is missing but on blocks Invoke and stays at full strength. */}
      <Box
        aria-label={name}
        opacity={isCompatibleModel && !(onWeightChangeEnd && !lora.isEnabled) ? 1 : 0.68}
        role="group"
      >
        <ListItem
          actions={
            <Flex align="center" gap={1}>
              {issue}
              <GenerateRowSwitch
                checked={isInstalled ? isActive : lora.isEnabled}
                disabled={!isCompatible}
                label={
                  isActive
                    ? t('widgets.generate.disableConcept', { name })
                    : t('widgets.generate.enableConcept', { name })
                }
                onCheckedChange={handleToggle}
              />
              <Tooltip content={t('widgets.generate.removeConcept')}>
                <IconButton
                  aria-label={t('widgets.generate.removeConceptNamed', { name })}
                  color="fg.muted"
                  size="sm"
                  variant="ghost"
                  onClick={handleRemoveClick}
                >
                  <Trash2Icon />
                </IconButton>
              </Tooltip>
            </Flex>
          }
          // The model type sits under the name, with the trigger phrases beside it on the same line.
          description={
            <Flex align="center" gap="1.5" minW="0">
              <Badge colorPalette={models.getBaseColorPalette(lora.model.base)} flexShrink={0} variant="surface">
                {models.getBaseLabel(lora.model.base)}
              </Badge>
              {isInstalled ? null : (
                <Badge colorPalette="red" flexShrink={0} variant="solid">
                  {t('widgets.generate.notInstalled')}
                </Badge>
              )}
              {isCompatibleModel ? null : (
                <Badge colorPalette="orange" flexShrink={0} variant="surface">
                  {t('widgets.generate.incompatible')}
                </Badge>
              )}
              {triggerPhrases?.length ? (
                <chakra.span minW="0" truncate>
                  {triggerPhrases.join(', ')}
                </chakra.span>
              ) : null}
            </Flex>
          }
          detail={
            // Backed with the widget's surface so the row's hover tint never washes out the track and fill.
            <Flex align="flex-start" bg="var(--wb-field-surface, var(--chakra-colors-bg-subtle))" rounded="control">
              <GenerateFieldContextMenu
                fieldLabel={t('widgets.generate.conceptWeightNamed', { name })}
                copyValue={copyWeight}
                isAtDefault={weight === defaultWeight}
                resetLabel={t('widgets.generate.resetToConceptDefault')}
                onReset={handleReset}
              >
                <ScrubberField
                  defaultValue={defaultWeight}
                  disabled={!isActive}
                  error={weightError}
                  hint="loraWeight"
                  inputMax={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax}
                  inputMin={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin}
                  label={t('widgets.generate.weight')}
                  marks={WEIGHT_MARKS}
                  max={DEFAULT_LORA_WEIGHT_CONFIG.sliderMax}
                  min={DEFAULT_LORA_WEIGHT_CONFIG.sliderMin}
                  step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
                  value={weight}
                  onChange={onPreviewWeight ?? handleWeightChange}
                  onChangeEnd={onWeightChangeEnd}
                />
              </GenerateFieldContextMenu>
              {onWeightChangeEnd ? (
                <Flex direction="column" flexShrink={0}>
                  <IconButton
                    aria-label={t('widgets.generate.increaseConceptWeight', { name })}
                    disabled={!isActive || weight >= DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax}
                    h="3.5"
                    minH="0"
                    minW="5"
                    size="xs"
                    variant="ghost"
                    onClick={handleIncrease}
                  >
                    <ChevronUpIcon />
                  </IconButton>
                  <IconButton
                    aria-label={t('widgets.generate.decreaseConceptWeight', { name })}
                    disabled={!isActive || weight <= DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin}
                    h="3.5"
                    minH="0"
                    minW="5"
                    size="xs"
                    variant="ghost"
                    onClick={handleDecrease}
                  >
                    <ChevronDownIcon />
                  </IconButton>
                </Flex>
              ) : null}
            </Flex>
          }
          density={compact ? 'compact' : 'snug'}
          leading={
            coverUrl ? (
              <ImagePreviewTooltip src={models.getImageUrl(key, undefined, 'preview')}>{thumbnail}</ImagePreviewTooltip>
            ) : (
              thumbnail
            )
          }
          isMenuOpen={menu.anchor !== null}
          role="presentation"
          title={name}
          onContextMenu={menu.open}
        />
      </Box>
      <Menu.Root
        lazyMount
        open={menu.anchor !== null}
        positioning={{
          getAnchorRect: () => (shownAnchor ? { height: 1, width: 1, x: shownAnchor.x, y: shownAnchor.y } : null),
          placement: 'bottom-start',
        }}
        unmountOnExit
        onExitComplete={releaseMenu}
        onOpenChange={(event) => {
          if (!event.open) {
            menu.close();
          }
        }}
        onRequestDismiss={menu.onRequestDismiss}
      >
        <Portal>
          <Menu.Positioner>
            {shownAnchor ? (
              <MenuContent minW="12rem">
                {openInModelManager ? (
                  <MenuActionItem
                    icon={ExternalLinkIcon}
                    label={t('widgets.generate.conceptMenu.openInModelManager')}
                    value="open-in-model-manager"
                    onSelect={() => openInModelManager(key)}
                  />
                ) : null}
                <MenuActionItem
                  disabled={weight === defaultWeight || !isActive}
                  icon={RotateCcwIcon}
                  label={t('widgets.generate.resetToConceptDefault')}
                  value="reset-weight"
                  onSelect={handleReset}
                />
                <MenuActionItem
                  disabled={!isCompatible}
                  icon={isActive ? PowerOffIcon : PowerIcon}
                  label={
                    isActive ? t('widgets.generate.conceptMenu.disable') : t('widgets.generate.conceptMenu.enable')
                  }
                  value="toggle"
                  onSelect={() => handleToggle(!isActive)}
                />
                <Menu.Separator />
                <MenuActionItem
                  icon={Trash2Icon}
                  label={t('widgets.generate.conceptMenu.remove')}
                  tone="danger"
                  value="remove"
                  onSelect={() => removeFrom(shownAnchor.focusTarget?.() ?? null)}
                />
              </MenuContent>
            ) : null}
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </Box>
  );
};
