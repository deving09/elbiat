"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import {
  LayoutDashboard,
  MessageSquare,
  MessageSquareText,
  Trophy,
  Image as ImageIcon,
  LogOut,
  Menu,
  X,
  Upload,
  ChevronDown,
  HelpCircle,
  MessageCircleQuestion,
} from "lucide-react";
import { useState, useRef, useEffect } from "react";

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Tasks", href: "/tasks", icon: Trophy },
  { name: "Gallery", href: "/gallery", icon: ImageIcon },
  { name: "Upload", href: "/upload", icon: Upload },  // Add this
  { name: "Feedback", href: "/feedback", icon: MessageSquareText },
];

const chatSubmenu = [
  { name: "Chat", href: "/chat", icon: MessageSquare },
  { name: "Ask", href: "/ask", icon: HelpCircle },
  { name: "Answer", href: "/answer", icon: MessageCircleQuestion },
];

export function Navbar() {
  const pathname = usePathname();
  const { user, logout, isAuthenticated } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [chatMenuOpen, setChatMenuOpen] = useState(false);
  const chatMenuRef = useRef<HTMLDivElement>(null);

  // Close chat dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (chatMenuRef.current && !chatMenuRef.current.contains(e.target as Node)) {
        setChatMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (!isAuthenticated) return null;

  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <div className="flex items-center">
            <Link href="/" className="flex items-center space-x-2">
              <img src="/logo.svg" alt="Taible" className="h-8 w-auto" />
              <span className="font-semibold text-lg hidden sm:block">Taible</span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex md:items-center md:space-x-1">
            {/* Dashboard link (first item) */}
            <Link
              href="/"
              className={cn(
                "flex items-center space-x-2 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                pathname === "/"
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              )}
            >
              <LayoutDashboard className="h-4 w-4" />
              <span>Dashboard</span>
            </Link>

            {/* Chat dropdown */}
            <div className="relative" ref={chatMenuRef}>
              <button
                className={cn(
                  "flex items-center space-x-2 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  pathname.startsWith("/chat") || pathname.startsWith("/ask") || pathname.startsWith("/answer")
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                )}
                onClick={() => setChatMenuOpen(!chatMenuOpen)}
                onDoubleClick={() => window.location.href = "/chat"}
              >
                <MessageSquare className="h-4 w-4" />
                <span>Chat</span>
                <ChevronDown className={cn("h-3 w-3 transition-transform", chatMenuOpen && "rotate-180")} />
              </button>

              {chatMenuOpen && (
                <div className="absolute top-full left-0 mt-1 bg-popover border rounded-lg shadow-lg py-1 min-w-[140px] z-50">
                  {chatSubmenu.map((item) => (
                    <Link
                      key={item.name}
                      href={item.href}
                      className={cn(
                        "flex items-center space-x-2 px-3 py-2 text-sm hover:bg-muted",
                        pathname === item.href && "bg-muted text-primary"
                      )}
                      onClick={() => setChatMenuOpen(false)}
                    >
                      <item.icon className="h-4 w-4" />
                      <span>{item.name}</span>
                    </Link>
                  ))}
                </div>
              )}
            </div>

            {/* Rest of navigation (skip Dashboard, already rendered) */}
            {navigation.slice(1).map((item) => {
              const isActive = pathname === item.href || 
                (item.href !== "/" && pathname.startsWith(item.href));
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "flex items-center space-x-2 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                  )}
                >
                  <item.icon className="h-4 w-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </div>

          {/* User section */}
          <div className="hidden md:flex md:items-center md:space-x-4">
            <span className="text-sm text-muted-foreground">
              {user?.username || user?.email}
            </span>
            <Button variant="ghost" size="sm" onClick={logout}>
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </div>

          {/* Mobile menu button */}
          <div className="flex md:hidden">
            <button
              type="button"
              className="inline-flex items-center justify-center rounded-md p-2 text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-border">
          <div className="space-y-1 px-4 py-3">
            {/* Dashboard */}
            <Link
              href="/"
              className={cn(
                "flex items-center space-x-3 rounded-md px-3 py-2 text-base font-medium",
                pathname === "/"
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-accent"
              )}
              onClick={() => setMobileMenuOpen(false)}
            >
              <LayoutDashboard className="h-5 w-5" />
              <span>Dashboard</span>
            </Link>

            {/* Chat submenu items */}
            {chatSubmenu.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "flex items-center space-x-3 rounded-md px-3 py-2 text-base font-medium",
                  pathname === item.href
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-accent"
                )}
                onClick={() => setMobileMenuOpen(false)}
              >
                <item.icon className="h-5 w-5" />
                <span>{item.name}</span>
              </Link>
            ))}

            {/* Rest of navigation (Tasks, Gallery, Upload, Feedback) */}
            {navigation.slice(1).map((item) => {
              const isActive = pathname === item.href ||
                (item.href !== "/" && pathname.startsWith(item.href));
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "flex items-center space-x-3 rounded-md px-3 py-2 text-base font-medium",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-accent"
                  )}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <item.icon className="h-5 w-5" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
            <div className="border-t border-border pt-3 mt-3">
              <div className="px-3 py-2 text-sm text-muted-foreground">
                {user?.username || user?.email}
              </div>
              <button
                className="flex w-full items-center space-x-3 rounded-md px-3 py-2 text-base font-medium text-muted-foreground hover:bg-accent"
                onClick={() => {
                  logout();
                  setMobileMenuOpen(false);
                }}
              >
                <LogOut className="h-5 w-5" />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}
