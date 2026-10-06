"use client";

import { useAuth } from "@/lib/auth";
import { UserRole } from "@/lib/api";
import { hasRole } from "@/lib/permissions";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

interface RoleGuardProps {
  children: React.ReactNode;
  requiredRole: UserRole;
  fallback?: React.ReactNode;
  redirectTo?: string;
}

export function RoleGuard({
  children,
  requiredRole,
  fallback = null,
  redirectTo,
}: RoleGuardProps) {
  const { user, isLoading, isAuthenticated } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !isAuthenticated && redirectTo) {
      router.push(redirectTo);
    }
  }, [isLoading, isAuthenticated, redirectTo, router]);

  if (isLoading) {
    return <>{fallback}</>;
  }

  if (!isAuthenticated || !user) {
    return <>{fallback}</>;
  }

  if (!hasRole(user.role, requiredRole)) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
}