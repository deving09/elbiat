import { UserRole } from "./api";

// Role hierarchy (higher index = more permissions)
const ROLE_HIERARCHY: UserRole[] = ["user", "enterprise", "admin"];

export function hasRole(userRole: UserRole, requiredRole: UserRole): boolean {
  const userLevel = ROLE_HIERARCHY.indexOf(userRole);
  const requiredLevel = ROLE_HIERARCHY.indexOf(requiredRole);
  return userLevel >= requiredLevel;
}

export function isAdmin(role: UserRole): boolean {
  return role === "admin";
}

export function isEnterprise(role: UserRole): boolean {
  return role === "enterprise" || role === "admin";
}

// Feature permissions
export const PERMISSIONS = {
  viewDemo: ["user", "enterprise", "admin"],
  manageUsers: ["admin"],
  viewAnalytics: ["enterprise", "admin"],
  accessAllExperiments: ["admin"],
  manageModels: ["admin"],
} as const;

export function canAccess(
  userRole: UserRole,
  permission: keyof typeof PERMISSIONS
): boolean {
  return (PERMISSIONS[permission] as readonly string[]).includes(userRole);
}