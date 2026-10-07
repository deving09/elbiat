export type FeedbackAction = "thumbs" | "edit" | "comment";

export interface FeedbackConfig {
  experimentBucket: string | null;
  availableActions: FeedbackAction[];
  features: {
    showThumbs: boolean;
    showEdit: boolean;
    showComment: boolean;
  };
}

const BUCKET_CONFIGS: Record<string, FeedbackAction[]> = {
  control: ["thumbs"],
  edit: ["thumbs", "edit"],
  comment: ["thumbs", "comment"],
  full: ["thumbs", "edit", "comment"],
};

const DEFAULT_ACTIONS: FeedbackAction[] = ["thumbs"];

export function getFeedbackConfig(experimentBucket: string | null): FeedbackConfig {
  const actions = experimentBucket && BUCKET_CONFIGS[experimentBucket]
    ? BUCKET_CONFIGS[experimentBucket]
    : DEFAULT_ACTIONS;

  return {
    experimentBucket,
    availableActions: actions,
    features: {
      showThumbs: actions.includes("thumbs"),
      showEdit: actions.includes("edit"),
      showComment: actions.includes("comment"),
    },
  };
}