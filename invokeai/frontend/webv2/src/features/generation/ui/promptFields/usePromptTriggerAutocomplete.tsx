import type { GenerateLora, GenerateModelConfig } from '@features/generation/core/types';
import type { CaretRect } from '@features/generation/ui/promptFields/promptCaret';
import type { PromptTriggerKey, PromptTriggerQuery } from '@features/generation/ui/promptFields/promptFocus';
import type { PromptTriggerOption } from '@features/generation/ui/promptFields/promptTriggerOptions';
import type { CompositionEvent, KeyboardEvent, ReactNode } from 'react';

import { getTextareaCaretRect } from '@features/generation/ui/promptFields/promptCaret';
import { getActiveTriggerQuery, insertPromptText } from '@features/generation/ui/promptFields/promptFocus';
import { PromptTriggerAutocomplete } from '@features/generation/ui/promptFields/PromptTriggerAutocomplete';
import {
  getInlineTriggerOptions,
  groupPromptTriggerOptions,
  usePromptTriggerOptions,
} from '@features/generation/ui/promptFields/promptTriggerOptions';
import { DismissOnViewportChange } from '@features/generation/ui/promptFields/useDismissOnViewportChange';
import { isImeComposing } from '@platform/browser/imeComposition';
import { useCallback, useId, useMemo, useState } from 'react';

const CARET_KEYS = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
const LIST_KEYS = ['ArrowUp', 'ArrowDown'];
const WILDCARD_KEYS = ['_'] as const;

interface AutocompleteState {
  angle?: { prefix: string; suffix: string };
  caretRect: CaretRect;
  projectId: string;
  query: PromptTriggerQuery;
  textarea: HTMLTextAreaElement;
  value: string;
}

export interface PromptTriggerAutocompleteApi {
  comboboxProps: {
    'aria-activedescendant': string | undefined;
    'aria-autocomplete': 'list';
    'aria-controls': string | undefined;
    'aria-expanded': boolean;
    role: 'combobox';
    onBlur: () => void;
    onCompositionEnd: (event: CompositionEvent<HTMLTextAreaElement>) => void;
    onCompositionStart: () => void;
    onKeyUp: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  };
  element: ReactNode;
  isOpen: boolean;
  refresh: (textarea: HTMLTextAreaElement | null) => void;
  handleKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
}

export const usePromptTriggerAutocomplete = ({
  isDisabled = false,
  keys,
  loras,
  onChange,
  projectId,
  selectedModel,
}: {
  keys: readonly PromptTriggerKey[];
  loras: GenerateLora[];
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  isDisabled?: boolean;
  onChange: (value: string) => void;
}): PromptTriggerAutocompleteApi => {
  const listboxId = useId();
  const optionIdPrefix = `${listboxId}-option-`;
  const [savedState, setState] = useState<AutocompleteState | null>(null);
  const [activeIndex, setActiveIndex] = useState<number | null>(0);
  const [lastAngleOption, setLastAngleOption] = useState<PromptTriggerOption | null>(null);
  const [isComposing, setIsComposing] = useState(false);
  const options = usePromptTriggerOptions(loras, selectedModel);

  const state = !isDisabled && savedState?.projectId === projectId ? savedState : null;
  const groups = useMemo(
    () =>
      state ? groupPromptTriggerOptions(getInlineTriggerOptions(options, state.query.key, state.query.query)) : [],
    [options, state]
  );
  const matches = useMemo(() => groups.flatMap((group) => group.options), [groups]);
  const isOpen = state !== null && !isComposing && matches.length > 0;
  const selectedIndex =
    activeIndex === null
      ? Math.max(
          0,
          matches.findIndex(
            (option) => option.group === lastAngleOption?.group && option.value === lastAngleOption.value
          )
        )
      : activeIndex < matches.length
        ? activeIndex
        : 0;
  const close = useCallback(() => setState(null), []);

  const refresh = useCallback(
    (textarea: HTMLTextAreaElement | null, finishedComposition = false) => {
      if (!textarea || isDisabled) {
        close();
        return;
      }
      if (isComposing && !finishedComposition) {
        return;
      }

      const { value, selectionStart, selectionEnd } = textarea;
      const wildcard = keys.includes('_') ? getActiveTriggerQuery(value, selectionStart, WILDCARD_KEYS) : null;

      if (state?.angle && state.textarea === textarea) {
        const { prefix, suffix } = state.angle;
        const start = prefix.length;
        const end = value.length - suffix.length;
        // An untouched selection is still what an immediate pick should replace.
        if (state.value === value && selectionStart === start && selectionEnd === state.query.range.end) {
          return;
        }
        if (!wildcard || wildcard.range.start < start) {
          const query = value.slice(start, end);
          // The shortcut leaves no delimiter in the prompt. Only edits between the saved boundaries belong to it.
          if (
            end >= start &&
            selectionStart === end &&
            selectionEnd === end &&
            value.startsWith(prefix) &&
            value.endsWith(suffix) &&
            !/[\n>]/.test(query) &&
            (state.value !== value || state.query.query === query)
          ) {
            if (state.value !== value) {
              setState({ ...state, query: { key: '<', query, range: { start, end } }, value });
              setActiveIndex(0);
            }
            return;
          }
        }
      }

      const caretRect = wildcard ? getTextareaCaretRect(textarea, wildcard.range.start) : null;
      setState(wildcard && caretRect ? { caretRect, projectId, query: wildcard, textarea, value } : null);
      setActiveIndex(0);
    },
    [close, isComposing, isDisabled, keys, projectId, state]
  );

  const selectOption = useCallback(
    (option: PromptTriggerOption) => {
      if (state && state.textarea.isConnected && state.textarea.value === state.value && !isComposing) {
        if (state.angle) {
          setLastAngleOption(option);
        }
        insertPromptText({
          onChange,
          range: state.query.range,
          text: option.value,
          textarea: state.textarea,
          value: state.textarea.value,
        });
      }

      setState(null);
    },
    [isComposing, onChange, state]
  );

  const handlePickerKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>): void => {
      if (
        !state ||
        (!isOpen && !state.angle) ||
        isComposing ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        isImeComposing(event.nativeEvent)
      ) {
        return;
      }

      if (CARET_KEYS.includes(event.key)) {
        return;
      }

      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        close();
        return;
      }

      if ((event.key === 'ArrowDown' || event.key === 'ArrowUp') && matches.length > 0) {
        const step = event.key === 'ArrowDown' ? 1 : -1;

        event.preventDefault();
        setActiveIndex((selectedIndex + step + matches.length) % matches.length);
        return;
      }

      if (event.key === 'Enter' || (event.key === 'Tab' && (!event.shiftKey || !state.angle))) {
        const option = matches[selectedIndex];

        if (option) {
          event.preventDefault();
          selectOption(option);
        }
      }
    },
    [close, isComposing, isOpen, matches, selectOption, selectedIndex, state]
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>): void => {
      if (event.key !== '<' || !keys.includes('<') || isDisabled || isComposing || isImeComposing(event.nativeEvent)) {
        handlePickerKeyDown(event);
        return;
      }

      const textarea = event.currentTarget;
      const range = { end: textarea.selectionEnd, start: textarea.selectionStart };
      const caretRect = getTextareaCaretRect(textarea, range.start);
      if (!caretRect) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      setActiveIndex(null);
      setState({
        angle: { prefix: textarea.value.slice(0, range.start), suffix: textarea.value.slice(range.end) },
        caretRect,
        projectId,
        query: { key: '<', query: '', range },
        textarea,
        value: textarea.value,
      });
    },
    [handlePickerKeyDown, isComposing, isDisabled, keys, projectId]
  );

  const handleKeyUp = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (CARET_KEYS.includes(event.key) || (!isOpen && LIST_KEYS.includes(event.key))) {
        refresh(event.currentTarget);
      }
    },
    [isOpen, refresh]
  );

  const handleCompositionStart = useCallback(() => {
    setIsComposing(true);
    if (!state?.angle) {
      close();
    }
  }, [close, state]);

  const handleCompositionEnd = useCallback(
    (event: CompositionEvent<HTMLTextAreaElement>) => {
      setIsComposing(false);
      refresh(event.currentTarget, true);
    },
    [refresh]
  );

  return {
    comboboxProps: {
      'aria-activedescendant': isOpen ? `${optionIdPrefix}${selectedIndex}` : undefined,
      'aria-autocomplete': 'list',
      'aria-controls': isOpen ? listboxId : undefined,
      'aria-expanded': isOpen,
      onBlur: close,
      onCompositionEnd: handleCompositionEnd,
      onCompositionStart: handleCompositionStart,
      onKeyUp: handleKeyUp,
      role: 'combobox',
    },
    element:
      isOpen && state ? (
        <>
          <DismissOnViewportChange dismiss={close} enabled />
          <PromptTriggerAutocomplete
            activeIndex={selectedIndex}
            caretRect={state.caretRect}
            groups={groups}
            listboxId={listboxId}
            optionIdPrefix={optionIdPrefix}
            onSelect={selectOption}
          />
        </>
      ) : null,
    handleKeyDown,
    isOpen,
    refresh,
  };
};
