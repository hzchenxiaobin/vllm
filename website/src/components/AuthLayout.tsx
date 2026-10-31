import { useAuth } from "@/hooks/useAuth";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import { LOGIN_PATH } from "@/const";
import { useIsMobile } from "@/hooks/use-mobile";
import { loadManifest, type WeekMeta } from "@/lib/content";
import {
  BookOpenCheck,
  Flame,
  LayoutDashboard,
  LogOut,
  MessageSquareText,
  PanelLeft,
} from "lucide-react";
import { type CSSProperties, type ReactNode, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { AuthLayoutSkeleton } from "./AuthLayoutSkeleton";
import { Button } from "./ui/button";

const menuItems = [
  { icon: LayoutDashboard, label: "仪表盘", path: "/" },
  { icon: Flame, label: "闪卡复习", path: "/review" },
  { icon: MessageSquareText, label: "模拟面试", path: "/interview" },
];

const SIDEBAR_WIDTH_KEY = "sidebar-width";
const DEFAULT_WIDTH = 280;
const MIN_WIDTH = 200;
const MAX_WIDTH = 480;

export default function AuthLayout({
  children,
}: {
  children: ReactNode;
}) {
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    const saved = localStorage.getItem(SIDEBAR_WIDTH_KEY);
    return saved ? parseInt(saved, 10) : DEFAULT_WIDTH;
  });
  const { isLoading, user } = useAuth();

  useEffect(() => {
    localStorage.setItem(SIDEBAR_WIDTH_KEY, sidebarWidth.toString());
  }, [sidebarWidth]);

  if (isLoading) {
    return <AuthLayoutSkeleton />;
  }

  if (!user) {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-background">
        <div className="flex w-full max-w-md flex-col items-center gap-8 p-8">
          <div className="flex flex-col items-center gap-4">
            <div className="font-tech text-xs uppercase tracking-[0.3em] text-primary">
              vLLM TRAINING CAMP
            </div>
            <h1 className="text-center text-2xl font-semibold tracking-tight">
              推理系统优化 · 学习训练营
            </h1>
            <p className="max-w-sm text-center text-sm text-muted-foreground">
              打卡 · 自测 · 闪卡 · 复盘 · 模拟面试，完成八周学习闭环。登录后进度多端同步。
            </p>
          </div>
          <Button
            onClick={() => {
              window.location.href = LOGIN_PATH;
            }}
            size="lg"
            className="w-full"
          >
            使用 Kimi 账号登录
          </Button>
        </div>
      </div>
    );
  }

  return (
    <SidebarProvider
      style={
        {
          "--sidebar-width": `${sidebarWidth}px`,
        } as CSSProperties
      }
    >
      <AuthLayoutContent setSidebarWidth={setSidebarWidth}>
        {children}
      </AuthLayoutContent>
    </SidebarProvider>
  );
}

type AuthLayoutContentProps = {
  children: ReactNode;
  setSidebarWidth: (width: number) => void;
};

/** 侧边栏周导航（W00-W09） */
function WeekNav({ collapsed }: { collapsed: boolean }) {
  const [weeks, setWeeks] = useState<WeekMeta[]>([]);
  const location = useLocation();
  const navigate = useNavigate();
  useEffect(() => {
    loadManifest()
      .then((m) => setWeeks(m.weeks))
      .catch(() => {});
  }, []);
  if (collapsed || weeks.length === 0) return null;
  return (
    <div className="mt-4 px-4">
      <div className="mb-2 flex items-center gap-2 font-tech text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
        <BookOpenCheck className="h-3 w-3" />
        学习路线
      </div>
      <div className="space-y-0.5">
        {weeks.map((w) => {
          const path = `/week/${w.week}`;
          const active = location.pathname === path;
          return (
            <button
              key={w.week}
              onClick={() => navigate(path)}
              className={`group flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                active ? "bg-sidebar-accent text-foreground" : "hover:bg-sidebar-accent/50"
              }`}
            >
              <span
                className={`font-tech text-[10px] ${
                  active ? "text-primary" : "text-muted-foreground group-hover:text-primary/70"
                }`}
              >
                W{String(w.week).padStart(2, "0")}
              </span>
              <span className="truncate text-xs">{w.title.replace(/^(第\s*\d+\s*周|Week\s*\d+)[：:]\s*/, "")}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function AuthLayoutContent({
  children,
  setSidebarWidth,
}: AuthLayoutContentProps) {
  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const { state, toggleSidebar } = useSidebar();
  const isCollapsed = state === "collapsed";
  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const activeMenuItem = menuItems.find(item => item.path === location.pathname);
  const isMobile = useIsMobile();

  useEffect(() => {
    if (isCollapsed) {
      setIsResizing(false);
    }
  }, [isCollapsed]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;

      const sidebarLeft = sidebarRef.current?.getBoundingClientRect().left ?? 0;
      const newWidth = e.clientX - sidebarLeft;
      if (newWidth >= MIN_WIDTH && newWidth <= MAX_WIDTH) {
        setSidebarWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isResizing, setSidebarWidth]);

  return (
    <>
      <div className="relative" ref={sidebarRef}>
        <Sidebar
          collapsible="icon"
          className="border-r-0"

        >
          <SidebarHeader className="h-16 justify-center">
            <div className="flex items-center gap-3 px-2 transition-all w-full">
              <button
                onClick={toggleSidebar}
                className="h-8 w-8 flex items-center justify-center hover:bg-accent rounded-lg transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring shrink-0"
                aria-label="Toggle navigation"
              >
                <PanelLeft className="h-4 w-4 text-muted-foreground" />
              </button>
              {!isCollapsed ? (
                <div className="flex min-w-0 flex-col">
                  <span className="truncate text-sm font-semibold tracking-tight">
                    vLLM 训练营
                  </span>
                  <span className="font-tech text-[10px] uppercase tracking-widest text-muted-foreground">
                    inference sys
                  </span>
                </div>
              ) : null}
            </div>
          </SidebarHeader>

          <SidebarContent className="gap-0">
            <SidebarMenu className="px-2 py-1">
              {menuItems.map(item => {
                const isActive = location.pathname === item.path;
                return (
                  <SidebarMenuItem key={item.path}>
                    <SidebarMenuButton
                      isActive={isActive}
                      onClick={() => navigate(item.path)}
                      tooltip={item.label}
                      className={`h-10 transition-all font-normal`}
                    >
                      <item.icon
                        className={`h-4 w-4 ${isActive ? "text-primary" : ""}`}
                      />
                      <span>{item.label}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
            <WeekNav collapsed={isCollapsed} />
          </SidebarContent>

          <SidebarFooter className="p-3">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="flex items-center gap-3 rounded-lg px-1 py-1 hover:bg-accent/50 transition-colors w-full text-left group-data-[collapsible=icon]:justify-center focus:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                  <Avatar className="h-9 w-9 border shrink-0">
                    <AvatarFallback className="text-xs font-medium">
                      {user?.name?.charAt(0).toUpperCase()}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0 group-data-[collapsible=icon]:hidden">
                    <p className="text-sm font-medium truncate leading-none">
                      {user?.name || "-"}
                    </p>
                    <p className="text-xs text-muted-foreground truncate mt-1.5">
                      {user?.email || "-"}
                    </p>
                  </div>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem
                  onClick={logout}
                  className="cursor-pointer text-destructive focus:text-destructive"
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>退出登录</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarFooter>
        </Sidebar>
        <div
          className={`absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-primary/20 transition-colors ${isCollapsed ? "hidden" : ""}`}
          onMouseDown={() => {
            if (isCollapsed) return;
            setIsResizing(true);
          }}
          style={{ zIndex: 50 }}
        />
      </div>

      <SidebarInset>
        {isMobile && (
          <div className="flex border-b h-14 items-center justify-between bg-background/95 px-2 backdrop-blur supports-backdrop-filter:backdrop-blur sticky top-0 z-40">
            <div className="flex items-center gap-2">
              <SidebarTrigger className="h-9 w-9 rounded-lg bg-background" />
              <div className="flex items-center gap-3">
                <div className="flex flex-col gap-1">
                  <span className="tracking-tight text-foreground">
                    {activeMenuItem?.label ?? "vLLM 训练营"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
        <main className="flex-1 p-4">{children}</main>
      </SidebarInset>
    </>
  );
}
