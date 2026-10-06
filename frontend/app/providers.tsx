"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { Navbar } from "@/components/layout/navbar";

const publicPaths = ["/", "/login", "/register"];  // Added "/"

function AuthGuard({ children }: { children: React.ReactNode }) {
  const { isLoading, isAuthenticated, checkAuth } = useAuth();
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  useEffect(() => {
    if (!isLoading) {
      const isPublicPath = publicPaths.includes(pathname);
      
      if (!isAuthenticated && !isPublicPath) {
        router.push("/login");
      } else if (isAuthenticated && pathname === "/login") {
        // Only redirect away from login page, not all public paths
        router.push("/dashboard");
      }
    }
  }, [isLoading, isAuthenticated, pathname, router]);

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="flex flex-col items-center space-y-4">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  const isPublicPath = publicPaths.includes(pathname);
  if (!isAuthenticated && !isPublicPath) {
    return null;
  }

  // Hide navbar on landing page
  const showNavbar = pathname !== "/";

  return (
    <>
      {showNavbar && <Navbar />}
      {children}
    </>
  );
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            retry: 1,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <AuthGuard>{children}</AuthGuard>
    </QueryClientProvider>
  );
}