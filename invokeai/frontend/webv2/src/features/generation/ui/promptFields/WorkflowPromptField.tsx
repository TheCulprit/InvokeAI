import type { GenerateLora } from '@features/generation/core/types';
import type { ChangeEvent } from 'react';

import { HStack, Stack, Text } from '@chakra-ui/react';
import {
  DYNAMIC_PROMPTS_DEFAULT_MAX_PROMPTS,
  type DynamicPromptsConfig,
} from '@features/generation/core/dynamicPrompts';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { useDynamicPrompts } from '@features/generation/ui/useDynamicPrompts';
import { useWildcards } from '@features/generation/ui/useWildcards';
import { Button } from '@platform/ui/Button';
import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowPromptTriggerScope } from './promptTriggerOptions';

import { AddPromptTriggerButton, PromptTriggerPopover } from './PositivePromptActions';
import { insertPromptText } from './promptFocus';
import { PromptTextarea } from './PromptTextarea';
import { usePromptTriggerAutocomplete } from './usePromptTriggerAutocomplete';
import { usePromptTriggerPicker } from './usePromptTriggerPicker';

const TRIGGER_KEYS = ['<', '_'] as const;
const NO_GENERATE_LORAS: GenerateLora[] = [];
const PREVIEW_CONFIG: DynamicPromptsConfig = {
  combinatorial: true,
  enabled: true,
  maxPrompts: DYNAMIC_PROMPTS_DEFAULT_MAX_PROMPTS,
  resample: false,
  sampleSeed: 0,
  seedBehaviour: 'per-iteration',
};

const WorkflowPromptPreview = ({ prompt }: { prompt: string }) => {
  const { t } = useTranslation();
  const preview = useDynamicPrompts(prompt, PREVIEW_CONFIG);

  return (
    <Stack
      gap="1"
      maxH="14rem"
      overflowY="auto"
      role="region"
      aria-label={t('widgets.generate.dynamicPrompts.promptsPreview')}
    >
      <Text fontSize="xs" color="fg.subtle">
        {t('widgets.workflow.formBuilder.promptPreviewHint')}
      </Text>
      {preview.isLoading ? (
        <Text role="status" fontSize="xs">
          {t('common.loading')}
        </Text>
      ) : preview.isError ? (
        <Button size="xs" variant="outline" loading={preview.isRetrying} onClick={preview.retry}>
          {t('common.retry')}
        </Button>
      ) : (
        <>
          {preview.error ? (
            <Text color="fg.error" fontSize="xs">
              {preview.error}
            </Text>
          ) : null}
          {preview.prompts.map((variation, index) => (
            <Text key={index} fontSize="xs" whiteSpace="pre-wrap" overflowWrap="anywhere">
              {variation}
            </Text>
          ))}
        </>
      )}
    </Stack>
  );
};

/** Edits the workflow's literal string. Preview calls Invoke's parser without running or modifying the workflow. */
export const WorkflowPromptField = ({
  id,
  invalid,
  label,
  onChange,
  projectId,
  scope,
  value,
}: {
  id: string;
  invalid: boolean;
  label: string;
  onChange: (value: string) => void;
  projectId: string;
  scope: WorkflowPromptTriggerScope;
  value: string;
}) => {
  const { t } = useTranslation();
  const { models, project } = useGenerationUi();
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const { knownNames } = useWildcards();
  const autocomplete = usePromptTriggerAutocomplete({
    keys: TRIGGER_KEYS,
    loras: NO_GENERATE_LORAS,
    onChange,
    projectId,
    selectedModel: undefined,
    workflowScope: scope,
  });
  const picker = usePromptTriggerPicker({
    insert: (text) => insertPromptText({ onChange, textarea: textareaRef.current, text, value }),
  });
  const handleChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(event.currentTarget.value);
    autocomplete.refresh(event.currentTarget);
  };

  return (
    <Stack gap="1" minW="0">
      <HStack gap="1">
        <AddPromptTriggerButton isOpen={picker.isOpen} onOpenPromptTriggerPicker={picker.open} />
        <Button size="xs" variant="ghost" aria-expanded={showPreview} onClick={() => setShowPreview(!showPreview)}>
          {t('widgets.generate.dynamicPrompts.promptsPreview')}
        </Button>
      </HStack>
      <PromptTextarea
        {...autocomplete.comboboxProps}
        aria-invalid={invalid || undefined}
        aria-label={label}
        attentionSyntax="none"
        className="nodrag nowheel"
        defaultHeightPx={96}
        highlightDynamicPrompts
        id={id}
        knownWildcards={knownNames}
        minHeightPx={56}
        resizeHandleAriaLabel={label}
        showSyntaxHighlighting={project.showPromptSyntaxHighlighting}
        textareaRef={textareaRef}
        value={value}
        w="full"
        onChange={handleChange}
        onClick={(event) => autocomplete.refresh(event.currentTarget)}
        onFocus={() => void models.ensureLoaded()}
        onKeyDown={autocomplete.handleKeyDown}
      />
      {autocomplete.element}
      {picker.dismissElement}
      {picker.isOpen ? (
        <PromptTriggerPopover
          loras={NO_GENERATE_LORAS}
          open
          positioning={picker.positioning}
          selectedModel={undefined}
          workflowScope={scope}
          onClose={picker.close}
          onSelect={picker.select}
        />
      ) : null}
      {showPreview ? <WorkflowPromptPreview prompt={value} /> : null}
    </Stack>
  );
};
