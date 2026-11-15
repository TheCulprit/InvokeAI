import type { InvocationTemplates, ProjectGraphState, WorkflowEdge } from './types';

import { isLoraFieldCollectionEntry, toLoraFieldCollectionList } from './fields';
import { isInvocationNode } from './types';

export const getPromptLoraLoaders = (graph: ProjectGraphState, templates: InvocationTemplates) =>
  graph.nodes.filter(isInvocationNode).flatMap((node) => {
    const template = templates[node.data.type];
    const loras = template?.inputs.loras;
    if (!template || loras?.type.name !== 'LoRAField' || !loras.uiModelBase?.length) {
      return [];
    }
    return [{ bases: loras.uiModelBase, label: node.data.label || template.title, node }];
  });

export const getPromptLoraScope = (
  graph: ProjectGraphState,
  templates: InvocationTemplates,
  loaderIds: readonly string[],
  edges: readonly WorkflowEdge[]
): { bases: string[]; selectedLoraKeys: string[] } => {
  const loaders = loaderIds.length
    ? getPromptLoraLoaders(graph, templates).filter(({ node }) => loaderIds.includes(node.id))
    : [];
  if (loaders.length === 0) {
    return { bases: [], selectedLoraKeys: [] };
  }

  const nodes = new Map(graph.nodes.filter(isInvocationNode).map((node) => [node.id, node]));
  const incoming = new Map<string, WorkflowEdge[]>();
  for (const edge of edges) {
    const existing = incoming.get(edge.target);
    if (existing) {
      existing.push(edge);
    } else {
      incoming.set(edge.target, [edge]);
    }
  }
  const from = (id: string, handles: readonly string[]) =>
    (incoming.get(id) ?? []).filter((edge) => handles.some((handle) => edge.targetHandle === handle));
  const keys = new Set<string>();
  const pending: WorkflowEdge[] = [];
  for (const { node } of loaders) {
    const inputs = from(node.id, ['loras']);
    pending.push(...inputs);
    if (!inputs.length) {
      for (const entry of toLoraFieldCollectionList(node.data.inputs.loras?.value)) {
        if (isLoraFieldCollectionEntry(entry) && entry.weight !== null && entry.weight !== 0) {
          keys.add(entry.lora.key);
        }
      }
    }
  }

  // Only follow nodes whose outputs can be read without executing the workflow.
  const seen = new Set<string>();
  while (pending.length) {
    const edge = pending.pop();
    const node = edge && nodes.get(edge.source);
    if (!edge || !node || seen.has(node.id)) {
      continue;
    }
    seen.add(node.id);
    if (node.data.type === 'collect' && edge.sourceHandle === 'collection') {
      pending.push(...from(node.id, ['item', 'collection']));
    }
    if (node.data.type === 'lora_selector' && edge.sourceHandle === 'lora' && !from(node.id, ['lora']).length) {
      const value = node.data.inputs.lora?.value;
      if (value && typeof value === 'object' && 'key' in value && typeof value.key === 'string') {
        if (node.data.inputs.weight?.value !== 0 || from(node.id, ['weight']).length) {
          keys.add(value.key);
        }
      }
    }
  }

  return { bases: [...new Set(loaders.flatMap(({ bases }) => bases))], selectedLoraKeys: [...keys] };
};
