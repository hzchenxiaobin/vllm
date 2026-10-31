import { Routes, Route } from 'react-router'
import Home from './pages/Home'
import Week from './pages/Week'
import Day from './pages/Day'
import Review from './pages/Review'
import Interview from './pages/Interview'
import Login from "./pages/Login"
import NotFound from "./pages/NotFound"

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/week/:week" element={<Week />} />
      <Route path="/day/:dayKey" element={<Day />} />
      <Route path="/review" element={<Review />} />
      <Route path="/interview" element={<Interview />} />
      <Route path="/login" element={<Login />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}
