import type { QueueActiveSession, QueueProgressSession } from '@features/queue/contracts';
import type { ReactNode } from 'react';

import {
  getAutoSwitchFollowedLive,
  getQueueActiveSessions,
  getQueueProgressSessions,
  isAutoSwitchPinHeld,
  isGalleryProgressItem,
  pickAutoSwitch,
} from '@features/queue/contracts';
import { useActiveProgressTargets, useFollowedProgressTargets, useFramedProgressTargets } from '@features/queue/react';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { createContext, use, useMemo, useState } from 'react';

interface LivePreviewFollow {
  sessions: QueueActiveSession[];
  gallerySessions: QueueProgressSession[];
  pinnedSessionId: string | null;
  /** The session Auto-switch shows (`getAutoSwitchFollowedLive`); else null, and the Preview keeps the selected result. */
  followedSessionId: string | null;
  /** The newest running session without a frame while none is followed: what Preview waits for. */
  startingSession: QueueActiveSession | null;
  /** Follows a chosen live tile; an unlocked mode changes to In progress. */
  follow(sessionId: string): void;
  pin(sessionId: string): void;
  showAll(): void;
  browse(): void;
}

const LivePreviewFollowContext = createContext<LivePreviewFollow | null>(null);

/** Owned by the mounted Workbench, never persisted with project documents. */
export const LivePreviewFollowProvider = ({ children }: { children: ReactNode }) => {
  const { projectId, items, mode, locked } = useActiveProjectSelector((project) => ({
    projectId: project.id,
    items: project.queue.items,
    mode: project.settings.viewerAutoSwitch,
    locked: project.settings.viewerAutoSwitchLocked === true,
  }));
  // Live frames show only on In progress; Newest switches to finished results alone.
  const enabled = mode === 'progress';
  const running = useActiveProgressTargets();
  const followed = useFollowedProgressTargets();
  const framed = useFramedProgressTargets();
  const sessions = useMemo(
    () => getQueueActiveSessions(items.filter(isGalleryProgressItem), running, followed, framed),
    [items, running, followed, framed]
  );
  const gallerySessions = useMemo(
    () => getQueueProgressSessions(items.filter(isGalleryProgressItem), sessions),
    [items, sessions]
  );
  const generationEvents = useMemo(
    () =>
      new Set([
        ...sessions.filter((session) => session.hasOwnFrame).map((session) => `frame:${session.id}`),
        ...items
          .filter(isGalleryProgressItem)
          .flatMap((item) => (item.completedBackendItemIds ?? []).map((id) => `result:${item.id}:${id}`)),
      ]),
    [items, sessions]
  );
  const [selection, setSelection] = useState({
    projectId,
    sessionId: null as string | null,
    browsing: false,
    manual: false,
    generationEvents,
  });
  const isStale =
    selection.projectId !== projectId ||
    ((selection.browsing || selection.manual) &&
      [...generationEvents].some((event) => !selection.generationEvents.has(event))) ||
    (selection.sessionId !== null &&
      (selection.manual
        ? !sessions.some((session) => session.id === selection.sessionId)
        : !enabled || !isAutoSwitchPinHeld(sessions, selection.sessionId)));
  if (isStale) {
    setSelection({ projectId, sessionId: null, browsing: false, manual: false, generationEvents });
  }
  const current = isStale ? null : selection;
  const pinnedSessionId = current?.sessionId ?? null;
  const followedSession = current?.browsing
    ? null
    : current?.manual && pinnedSessionId
      ? (sessions.find((session) => session.id === pinnedSessionId) ?? null)
      : getAutoSwitchFollowedLive(mode, sessions, pinnedSessionId);
  const followedSessionId = followedSession?.id ?? null;
  const startingSession =
    !current?.browsing && enabled && followedSession?.state !== 'running'
      ? (sessions.filter((session) => session.state === 'running' && !session.hasOwnFrame).at(-1) ?? null)
      : null;
  const { account } = useWorkbenchCommands();
  const value = useMemo<LivePreviewFollow>(
    () => ({
      sessions,
      gallerySessions,
      pinnedSessionId,
      followedSessionId,
      startingSession,
      follow: (sessionId) => {
        const { mode: nextMode, pinnedId } = pickAutoSwitch({ id: sessionId, kind: 'live' });
        if (!locked) {
          account.updateProjectPreferences({ viewerAutoSwitch: nextMode });
        }
        setSelection({ projectId, sessionId: pinnedId, browsing: false, manual: locked, generationEvents });
      },
      pin: (sessionId) => setSelection({ projectId, sessionId, browsing: false, manual: locked, generationEvents }),
      showAll: () => setSelection({ projectId, sessionId: null, browsing: false, manual: false, generationEvents }),
      browse: () => {
        if (locked) {
          setSelection({ projectId, sessionId: null, browsing: true, manual: false, generationEvents });
        }
      },
    }),
    [
      account,
      generationEvents,
      locked,
      sessions,
      gallerySessions,
      followedSessionId,
      pinnedSessionId,
      projectId,
      startingSession,
    ]
  );
  return <LivePreviewFollowContext value={value}>{children}</LivePreviewFollowContext>;
};

export const useLivePreviewFollow = (): LivePreviewFollow => {
  const context = use(LivePreviewFollowContext);
  if (!context) {
    throw new Error('Live preview requires its Workbench provider.');
  }
  return context;
};
